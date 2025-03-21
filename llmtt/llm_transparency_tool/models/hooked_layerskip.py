import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformer_lens import HookedTransformer, get_model_config


class HookedLayerSkip(HookedTransformer):
    """
    A custom wrapper for the LayerSkip model loaded from a Hugging Face checkpoint.
    It adds early-exit functionality via an assistant model and integrates with the
    llm-transparency-tool by registering hooks for introspection.
    
    Parameters:
      - exit_layer: the index of the layer at which to early exit.
      - num_speculations: number of speculative generations (passed for completeness).
      - checkpoint: the Hugging Face checkpoint name (e.g. "facebook/layerskip-llama3.2-1B").
    """
    def __init__(self, config, exit_layer=None, num_speculations=None, checkpoint=None):
        # Initialize the base HookedTransformer to get introspection capabilities.
        super().__init__(config)
        self.exit_layer = exit_layer
        self.num_speculations = num_speculations
        
        # Load the model from Hugging Face.
        # The checkpoint should be something like "facebook/layerskip-llama3.2-1B"
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            device_map="auto",
            use_safetensors=True,
            torch_dtype=torch.bfloat16
        )
        
        # Save the generation config for later use.
        self.generation_config = self.model.generation_config
        self.generation_config.__dict__['transformers_version'] = '4.45.0'

    @classmethod
    def from_pretrained(cls, model_name_or_path, **kwargs):
        """
        Loads a pretrained LayerSkip model (from its Hugging Face checkpoint) and wraps it
        for use with the llm-transparency-tool.
        
        Additional kwargs:
          - exit_layer: for early exit.
          - num_speculations: for speculative generation.
        """
        config = get_model_config(model_name_or_path)
        exit_layer = kwargs.pop("exit_layer", None)
        num_speculations = kwargs.pop("num_speculations", None)
        # Here model_name_or_path is assumed to be the Hugging Face checkpoint name.
        return cls(config, exit_layer=exit_layer, num_speculations=num_speculations, checkpoint=model_name_or_path)

    def forward(self, x):
        """
        Runs a forward pass through the model while registering hooks to capture intermediate
        activations. If an exit_layer is specified, hooks are only attached to layers up to that layer.
        """
        # Clear previous activations.
        self.activations = {}

        def save_activation(name):
            def hook(module, input, output):
                self.activations[name] = output
            return hook

        # Locate the transformer submodule.
        # For Llama-based models loaded via Hugging Face, the transformer body is typically in self.model.model.
        if hasattr(self.model, "model"):
            transformer = self.model.model
        else:
            transformer = self.model

        layers = getattr(transformer, "layers", None)
        if layers is None:
            raise ValueError("Could not locate transformer layers for hooking.")
        
        # Register hooks on each layer up to the early exit layer.
        for idx, layer in enumerate(layers):
            if self.exit_layer is not None and idx >= self.exit_layer:
                break
            layer.register_forward_hook(save_activation(f"layer_{idx}"))
        
        # Run the forward pass.
        output = self.model(x)
        return output

    def create_assistant_model(self):
        """
        Creates an assistant model for early exit by deep copying the main model (with shared weights)
        and truncating its transformer layers up to self.exit_layer.
        """
        weights_memo = {id(w): w for w in self.parameters()}
        assistant_model = deepcopy(self, memo=weights_memo)
        # Then truncate the layers as before:
        
        if self.exit_layer is not None:
            if hasattr(assistant_model, "model"):
                assistant_model.model.layers = assistant_model.model.layers[:self.exit_layer]
            else:
                assistant_model.layers = assistant_model.layers[:self.exit_layer]
        return assistant_model


    def generate(self, **kwargs):
        """
        Overrides the generate method. If an early exit is specified, an assistant model is created
        (with truncated layers) and passed via the assistant_model keyword argument.
        """
        if self.exit_layer is not None:
            kwargs["assistant_model"] = self.create_assistant_model()
        return self.model.generate(**kwargs)
    
    def run_with_cache(self, *args, **kwargs):
        """
        Overrides the default run_with_cache behavior to use the assistant (early-exit)
        model if an exit_layer is set.

        This ensures that when you run analysis (via run() which calls run_with_cache)
        on the model, it uses the truncated assistant model and captures activations from
        only the layers up to self.exit_layer.
        """
        if self.exit_layer is not None:
            assistant_model = self.create_assistant_model()
            return assistant_model.run_with_cache(*args, **kwargs)
        else:
            return super().run_with_cache(*args, **kwargs)

