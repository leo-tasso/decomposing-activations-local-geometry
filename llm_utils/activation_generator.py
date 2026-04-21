# Standard library imports
from types import SimpleNamespace
from typing import List, Tuple, Union, Callable
from collections import Counter

# Third-party imports
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import BitsAndBytesConfig
except Exception:  # pragma: no cover - optional quantization support
    BitsAndBytesConfig = None

# Project-specific imports
try:
    from transformer_lens import HookedTransformer, utils
except Exception:  # pragma: no cover - optional backend
    HookedTransformer = None
    utils = None
from data_utils.concept_dataset import ConceptDataset, SupervisedConceptDataset


class _HFModelAdapter:
    def __init__(self, model_name: str, device: str):
        self.model_name = model_name
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        model_kwargs = {
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
        }
        dtype = torch.float16 if str(device).startswith("cuda") else torch.float32

        use_4bit = False
        if str(device).startswith("cuda") and BitsAndBytesConfig is not None:
            try:
                import bitsandbytes  # noqa: F401
                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model_kwargs["device_map"] = "auto"
                model_kwargs["torch_dtype"] = torch.float16
                use_4bit = True
            except Exception:
                model_kwargs["torch_dtype"] = dtype
        else:
            model_kwargs["torch_dtype"] = dtype

        self._model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
        if not use_4bit:
            self._model = self._model.to(device)
        self._model.eval()

        n_layers = getattr(self._model.config, "num_hidden_layers", None)
        if n_layers is None:
            n_layers = getattr(self._model.config, "n_layer", None)
        d_model = getattr(self._model.config, "hidden_size", None)
        if d_model is None:
            d_model = getattr(self._model.config, "n_embd", None)

        self.cfg = SimpleNamespace(
            n_layers=int(n_layers) if n_layers is not None else 0,
            d_model=int(d_model) if d_model is not None else 0,
            device=device,
        )

    def to_tokens(self, prompts, padding_side: str = "left"):
        single_prompt = isinstance(prompts, str)
        prompt_list = [prompts] if single_prompt else prompts
        old_side = self.tokenizer.padding_side
        self.tokenizer.padding_side = padding_side
        enc = self.tokenizer(prompt_list, return_tensors="pt", padding=True)
        self.tokenizer.padding_side = old_side
        return enc["input_ids"]

    def to_string(self, token_or_tokens):
        if isinstance(token_or_tokens, torch.Tensor):
            token_or_tokens = token_or_tokens.detach().cpu().tolist()
        if isinstance(token_or_tokens, int):
            token_or_tokens = [token_or_tokens]
        return self.tokenizer.decode(token_or_tokens, skip_special_tokens=False)

    def to_str_tokens(self, token_ids):
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.detach().cpu().tolist()
        return [self.tokenizer.decode([int(tok)], skip_special_tokens=False) for tok in token_ids]

    def __call__(self, input_ids):
        with torch.no_grad():
            out = self._model(input_ids=input_ids.to(self.device), use_cache=False)
        return out.logits

    def reset_hooks(self):
        return


class ActivationGenerator:
    def __init__(
        self,
        model_name: str,
        model_device: str = "cpu",
        data_device: str = "cpu",
        mode: str = "residual",
        backend: str = "auto",
    ):
        """
        Initialize the generator with a pretrained model.
        
        Args:
            model_name (str): Name of the pretrained model.
            model_device (str): Device to load the model onto.
            data_device (str): Device to load the data onto.
            mode (str): Which activation to use ("mlp" or "residual").
            backend (str): "auto", "transformer_lens", or "hf".
        """
        self.backend = backend
        self.model_device = model_device

        if backend == "transformer_lens":
            if HookedTransformer is None:
                raise ImportError("transformer_lens is not available for backend='transformer_lens'")
            self.model = HookedTransformer.from_pretrained(model_name, device=model_device)
            self.backend = "transformer_lens"
        elif backend == "hf":
            self.model = _HFModelAdapter(model_name, model_device)
            self.backend = "hf"
        else:
            if HookedTransformer is not None:
                try:
                    self.model = HookedTransformer.from_pretrained(model_name, device=model_device)
                    self.backend = "transformer_lens"
                except Exception:
                    self.model = _HFModelAdapter(model_name, model_device)
                    self.backend = "hf"
            else:
                self.model = _HFModelAdapter(model_name, model_device)
                self.backend = "hf"

        self.model_name = model_name  # store for later use in helper functions
        self.data_device = data_device
        self._mode = mode

    def _token_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        pad_token_id = self.model.tokenizer.pad_token_id
        bos_token_id = self.model.tokenizer.bos_token_id
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        if pad_token_id is not None:
            mask = mask & (input_ids != pad_token_id)
        if bos_token_id is not None:
            mask = mask & (input_ids != bos_token_id)
        return mask

    def _get_data_as_tensors(self, dataset: ConceptDataset, batch_size: int):
        """
        Converts data from the ConceptDataset into model-ready tensors.
        Assumes that the dataset yields (prompts, labels) and uses left padding.
        """
        data = []
        for batch in dataset.get_batches(batch_size=batch_size):
            prompts = batch['prompt']
            tokens = self.model.to_tokens(prompts)
            data.append(tokens)
        return data

    def _get_mlp_hook_string(self, layer_number: int) -> str:
        """
        Helper to get the hook string for a given layer based on the mode.
        """
        if self.backend == "hf":
            if self._mode == "residual":
                return f"hf.hidden_states.{layer_number + 1}"
            if self._mode == "residual_pre":
                return f"hf.hidden_states.{layer_number}"
            raise NotImplementedError(
                "HF backend currently supports only 'residual' and 'residual_pre' modes."
            )

        if self._mode == 'mlp':
            act_str = f"blocks.{layer_number}.mlp.hook_post"
        if self._mode == "mlp_out":
            return f"blocks.{layer_number}.hook_mlp_out"
        elif self._mode == 'residual':
            if utils is None:
                return f"blocks.{layer_number}.hook_resid_post"
            act_str = utils.get_act_name("resid_post", layer_number)
        elif self._mode == 'residual_pre':
            if utils is None:
                return f"blocks.{layer_number}.hook_resid_pre"
            act_str = utils.get_act_name("resid_pre", layer_number)
        elif self._mode == 'attn_out':
            act_str = f"blocks.{layer_number}.hook_attn_out"
        return act_str

    def _hidden_states_for_input(self, input_ids: torch.Tensor):
        with torch.no_grad():
            outputs = self.model._model(
                input_ids=input_ids.to(self.model_device),
                output_hidden_states=True,
                use_cache=False,
            )
        return outputs.hidden_states

    def _hf_hidden_state_index(self, layer: int) -> int:
        if self._mode == "residual":
            return layer + 1
        if self._mode == "residual_pre":
            return layer
        raise NotImplementedError("HF backend supports only residual/residual_pre activations.")

    def change_mode(self, new_mode):
        self._mode = new_mode

    def _get_period_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Given a tensor of input IDs, returns a boolean mask where True corresponds
        to positions that match the period token (".").
        """
        # Assume that encoding "." returns a list with the period token id as its first element.
        period_id = self.model.tokenizer.encode(".")[0]
        return input_ids == period_id

    def build_vocab_frequency(self, dataset: ConceptDataset, batch_size: int = 5) -> Counter:
        """
        Builds a vocabulary frequency Counter over the entire dataset.
        Padding tokens are ignored.
        
        Args:
            dataset (ConceptDataset): The dataset yielding samples.
            batch_size (int): Batch size for processing.
            
        Returns:
            A Counter mapping token id to frequency count.
        """
        token_counter = Counter()
        data = self._get_data_as_tensors(dataset, batch_size)
        pad_token_id = self.model.tokenizer.pad_token_id
        for batch in tqdm(data, desc="Building vocab frequency"):
            # Supports either dict or tensor
            if isinstance(batch, dict):
                input_ids = batch["input_ids"]
            else:
                input_ids = batch
            tokens = input_ids.flatten().tolist()
            tokens = [t for t in tokens if t != pad_token_id]
            token_counter.update(tokens)
        return token_counter

    def generate_multiple_layer_activations_and_freq(
        self,
        dataset: Union[ConceptDataset, SupervisedConceptDataset],
        layers: List[int],
        batch_size: int = 5,
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        For each sample in the dataset, returns the activations from multiple layers
        and a frequency vector corresponding to each non-padding token.
        
        The output for each layer is a tensor of shape:
            (num_tokens, d_model)
        where num_tokens is the total number of non-padding tokens across the dataset.
        The frequency vector (of shape (num_tokens,)) is built from a vocabulary frequency
        computed over the dataset.
        
        Args:
            dataset (ConceptDataset): Dataset yielding samples.
            layers (List[int]): List of layer numbers to extract activations from.
            batch_size (int): Batch size for processing the dataset.
        
        Returns:
            A tuple (final_activations, freq) where:
              - final_activations: List of tensors, one per layer, each of shape (num_tokens, d_model).
              - freq: Tensor of shape (num_tokens,), where each entry is the frequency of that token.
        """
        # Build the global vocabulary frequency dictionary.
        vocab_freq = self.build_vocab_frequency(dataset, batch_size=batch_size)
        
        data = self._get_data_as_tensors(dataset, batch_size)
        all_layer_activations = [[] for _ in layers]
        all_token_ids = []

        with torch.no_grad():
            for batch in tqdm(data, desc="Generating multi-layer activations with freq"):
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.data_device) for k, v in batch.items()}
                    input_ids = inputs["input_ids"]
                else:
                    input_ids = batch.to(self.data_device)
                    inputs = None

                mask = self._token_mask(input_ids)
                if self.backend == "transformer_lens":
                    _, cache = self.model.run_with_cache(input_ids)
                else:
                    hidden_states = self._hidden_states_for_input(input_ids)

                # Extract non-padding token IDs.
                nonpad_ids = input_ids[mask.bool()].view(-1)
                all_token_ids.append(nonpad_ids.cpu())
                
                for idx, layer in enumerate(layers):
                    if self.backend == "transformer_lens":
                        hook_str = self._get_mlp_hook_string(layer)
                        acts = cache[hook_str].detach().to(self.data_device)
                    else:
                        hs_idx = self._hf_hidden_state_index(layer)
                        acts = hidden_states[hs_idx].detach().to(self.data_device)
                    # Extract only non-padding activations.
                    nonpad_acts = acts[mask.bool()].view(-1, acts.size(-1))
                    # Immediately move activations to CPU.
                    all_layer_activations[idx].append(nonpad_acts.cpu())
                
                if self.backend == "transformer_lens":
                    del cache
                else:
                    del hidden_states
                torch.cuda.empty_cache()
        
        # Concatenate activations for each layer and token IDs.
        final_activations = [torch.cat(layer_acts, dim=0) for layer_acts in all_layer_activations]
        token_ids_all = torch.cat(all_token_ids, dim=0)
        # Build the frequency vector: for each token in token_ids_all, look up its global frequency.
        freq = torch.tensor([vocab_freq[token.item()] for token in token_ids_all])
        self.model.reset_hooks()
        return final_activations, freq

    def generate_stacked_layer_activations_and_freq(
        self,
        dataset: Union[ConceptDataset, SupervisedConceptDataset],
        layers: List[int],
        batch_size: int = 5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For each sample in the dataset, returns the stacked activations from multiple layers
        concatenated along the feature dimension, along with a frequency vector corresponding
        to each non-padding token.

        The output activations tensor will have shape:
            (num_tokens, num_layers * d_model)
        where num_tokens is the total number of non-padding tokens across the dataset.
        The frequency vector (of shape (num_tokens,)) is built from a vocabulary frequency
        computed over the dataset.

        Args:
            dataset (ConceptDataset): Dataset yielding samples.
            layers (List[int]): List of layer numbers to extract activations from.
            batch_size (int): Batch size for processing the dataset.

        Returns:
            A tuple (stacked_activations, freq) where:
            - stacked_activations: Tensor of shape (num_tokens, num_layers * d_model)
                containing activations from all layers concatenated along the feature dimension.
            - freq: Tensor of shape (num_tokens,), where each entry is the frequency of that token.
        """
        # Build the global vocabulary frequency dictionary.
        vocab_freq = self.build_vocab_frequency(dataset, batch_size=batch_size)
        
        data = self._get_data_as_tensors(dataset, batch_size)
        # Initialize lists to collect activations for each layer.
        all_layer_activations = [[] for _ in layers]
        all_token_ids = []

        with torch.no_grad():
            for batch in tqdm(data, desc="Generating stacked activations with freq"):
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.data_device) for k, v in batch.items()}
                    input_ids = inputs["input_ids"]
                else:
                    input_ids = batch.to(self.data_device)
                    inputs = None

                mask = self._token_mask(input_ids)
                if self.backend == "transformer_lens":
                    _, cache = self.model.run_with_cache(input_ids)
                else:
                    hidden_states = self._hidden_states_for_input(input_ids)

                # Extract non-padding token IDs.
                nonpad_ids = input_ids[mask.bool()].view(-1)
                all_token_ids.append(nonpad_ids.cpu())
                
                for idx, layer in enumerate(layers):
                    if self.backend == "transformer_lens":
                        hook_str = self._get_mlp_hook_string(layer)
                        acts = cache[hook_str].detach().to(self.data_device)
                    else:
                        hs_idx = self._hf_hidden_state_index(layer)
                        acts = hidden_states[hs_idx].detach().to(self.data_device)
                    # Extract only non-padding activations.
                    nonpad_acts = acts[mask.bool()].view(-1, acts.size(-1))
                    all_layer_activations[idx].append(nonpad_acts.cpu())
                
                if self.backend == "transformer_lens":
                    del cache
                else:
                    del hidden_states
                torch.cuda.empty_cache()
        
        # Concatenate activations for each layer.
        final_activations = [torch.cat(layer_acts, dim=0) for layer_acts in all_layer_activations]
        # Stack activations from all layers along the feature dimension.
        stacked_activations = torch.cat(final_activations, dim=1)
        token_ids_all = torch.cat(all_token_ids, dim=0)
        # Build the frequency vector: for each token in token_ids_all, look up its global frequency.
        freq = torch.tensor([vocab_freq[token.item()] for token in token_ids_all])
        self.model.reset_hooks()
        return stacked_activations, freq


    def generate_period_activations(
        self,
        dataset: Union[ConceptDataset, SupervisedConceptDataset],
        layers: List[int],
        batch_size: int = 5,
    ) -> List[torch.Tensor]:
        """
        For each sample in the dataset, returns the activations corresponding to period tokens (".")
        from multiple layers.
        
        The output for each layer is a tensor of shape:
            (num_period_tokens, d_model)
        where num_period_tokens is the total number of period tokens across the dataset.
        
        Args:
            dataset (ConceptDataset): Dataset yielding samples.
            layers (List[int]): List of layer numbers to extract activations from.
            batch_size (int): Batch size for processing the dataset.
        
        Returns:
            A list of tensors, one per layer, each of shape (num_period_tokens, d_model),
            corresponding to activations for period tokens.
        """
        data = self._get_data_as_tensors(dataset, batch_size)
        period_layer_activations = [[] for _ in layers]
        
        with torch.no_grad():
            for batch in tqdm(data, desc="Generating period activations"):
                if isinstance(batch, dict):
                    inputs = {k: v.to(self.data_device) for k, v in batch.items()}
                    input_ids = inputs["input_ids"]
                else:
                    input_ids = batch.to(self.data_device)
                    inputs = None

                if self.backend == "transformer_lens":
                    _, cache = self.model.run_with_cache(input_ids)
                else:
                    hidden_states = self._hidden_states_for_input(input_ids)
                
                # Create a mask for period tokens.
                period_mask = self._get_period_mask(input_ids)
                
                for idx, layer in enumerate(layers):
                    if self.backend == "transformer_lens":
                        hook_str = self._get_mlp_hook_string(layer)
                        acts = cache[hook_str].detach().to(self.data_device)
                    else:
                        hs_idx = self._hf_hidden_state_index(layer)
                        acts = hidden_states[hs_idx].detach().to(self.data_device)
                    # Extract activations corresponding to period tokens.
                    period_acts = acts[period_mask.bool()].view(-1, acts.size(-1))
                    period_layer_activations[idx].append(period_acts.cpu())
                
                if self.backend == "transformer_lens":
                    del cache
                else:
                    del hidden_states
                torch.cuda.empty_cache()
        
        final_period_activations = [torch.cat(layer_acts, dim=0) for layer_acts in period_layer_activations]
        return final_period_activations



def extract_token_ids_sample_ids_and_labels(dataset: ConceptDataset, act_generator: ActivationGenerator, batch_size: int = 5):
    """
    Efficiently extract non-padding token IDs and corresponding labels from a dataset using the provided
    act_generator's tokenizer (without running the model or extracting activations).

    Args:
        dataset (ConceptDataset): A dataset instance that yields batches with at least a "prompt" key.
        act_generator (ActivationGenerator): Instance with a model containing a tokenizer and data_device.
        batch_size (int): Batch size for processing the dataset.

    Returns:
        token_ids (torch.Tensor): Tensor of shape (num_tokens,) containing the token IDs
                                  for all non-padding tokens in the dataset.
        labels (List): List of labels corresponding to each non-padding token.
    """
    all_token_ids = []
    all_labels = []
    sample_ids = []
    pad_token_id = act_generator.model.tokenizer.pad_token_id
    idx = 0

    for batch in tqdm(dataset.get_batches(batch_size=batch_size), desc="Extracting token IDs"):
        prompts = batch['prompt']
        labels = batch['label']
        
        # Tokenize the prompts (using left padding to be consistent)
        tokens = act_generator.model.to_tokens(prompts, padding_side="left")
        

        input_ids = tokens.to(act_generator.data_device)
        pad_token_id = act_generator.model.tokenizer.pad_token_id
        bos_token_id = act_generator.model.tokenizer.bos_token_id
        attention_mask = (input_ids != pad_token_id) & (input_ids != bos_token_id)
        
        # Count non-padding tokens per sample and repeat labels accordingly
        num_non_padding = attention_mask.sum(dim=1).squeeze()
        for n, label in zip(num_non_padding, labels):
            all_labels += [label] * n
            sample_ids += [idx] * n
            idx += 1
        
        # Filter out pad tokens and collect token IDs
        nonpad_ids = input_ids[attention_mask].view(-1)
        all_token_ids.append(nonpad_ids.cpu())

    token_ids = torch.cat(all_token_ids, dim=0)
    return token_ids, sample_ids, all_labels


import torch
from tqdm import tqdm

def extract_token_ids_and_sample_ids(dataset: ConceptDataset, act_generator: ActivationGenerator, batch_size: int = 5):
    """
    Efficiently extract non-padding token IDs and sample IDs from a dataset using the provided
    act_generator's tokenizer (without running the model or extracting activations).

    Args:
        dataset (ConceptDataset): A dataset instance that yields batches with at least a "prompt" key.
        act_generator (ActivationGenerator): Instance with a model containing a tokenizer and data_device.
        batch_size (int): Batch size for processing the dataset.

    Returns:
        token_ids (torch.Tensor): Tensor of shape (num_tokens,) containing the token IDs
                                  for all non-padding tokens in the dataset.
        sample_ids (List[int]): List of sample IDs corresponding to each non-padding token.
    """
    all_token_ids = []
    sample_ids = []
    pad_token_id = act_generator.model.tokenizer.pad_token_id
    bos_token_id = act_generator.model.tokenizer.bos_token_id
    idx = 0

    for batch in tqdm(dataset.get_batches(batch_size=batch_size), desc="Extracting token IDs"):
        prompts = batch['prompt']

        # Tokenize the prompts (using left padding to be consistent)
        tokens = act_generator.model.to_tokens(prompts, padding_side="left")
        input_ids = tokens.to(act_generator.data_device)

        # Create attention mask ignoring PAD and BOS
        attention_mask = (input_ids != pad_token_id) & (input_ids != bos_token_id)

        # Count non-padding tokens per sample and repeat sample IDs accordingly
        num_non_padding = attention_mask.sum(dim=1).squeeze()
        for n in num_non_padding:
            sample_ids += [idx] * n
            idx += 1

        # Filter out pad tokens and collect token IDs
        nonpad_ids = input_ids[attention_mask].view(-1)
        all_token_ids.append(nonpad_ids.cpu())

    token_ids = torch.cat(all_token_ids, dim=0)
    return token_ids, sample_ids
