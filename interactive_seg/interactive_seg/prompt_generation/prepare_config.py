
from interactive_seg.config.prompt_generation_config import NNInteractivePromptGenerationConfigBase
from interactive_seg_nninteractive_v1.config import NNInteractiveV1ModelConfig
from interactive_seg_nninteractive_v2.config import NNInteractiveV2ModelConfig

def prepare_nninteractive_config(
    config: NNInteractivePromptGenerationConfigBase | dict,
    model_config: NNInteractiveV1ModelConfig | NNInteractiveV2ModelConfig | dict
) -> tuple[NNInteractivePromptGenerationConfigBase, NNInteractiveV1ModelConfig | NNInteractiveV2ModelConfig]:
    """
    Validate the prompt generation configuration and model configuration.

    Args:
        config: An instance of NNInteractivePromptGenerationConfigBase or a dictionary containing the prompt 
            generation configuration.
        model_config: An instance of NNInteractiveV1ModelConfig, NNInteractiveV2ModelConfig, or a dictionary 
            containing the model configuration.

    Returns:
        A tuple containing the validated prompt generation configuration and model configuration.

    Raises:
        ValueError: If the config or model_config is not of the expected type or if ROI cropping is enabled 
            for model version v1.
    """
    # Validate the config if it is provided as a dictionary and convert it to an instance of NNInteractivePromptGenerationConfig
    if isinstance(config, dict):
        config = NNInteractivePromptGenerationConfigBase.model_validate(config)
    elif not isinstance(config, NNInteractivePromptGenerationConfigBase):
        raise ValueError("Config must be an instance of NNInteractivePromptGenerationConfigBase or a dictionary.")
    
    # Validate the model_config if it is provided as a dictionary and convert it to an instance of the appropriate model config class
    if isinstance(model_config, dict):
        if model_config.get("model_version") == "v1":
            model_config = NNInteractiveV1ModelConfig.model_validate(model_config)
        elif model_config.get("model_version") == "v2":
            model_config = NNInteractiveV2ModelConfig.model_validate(model_config)
        else:
            raise ValueError("model_config must be an instance of NNInteractiveV1ModelConfig or NNInteractiveV2ModelConfig or a dictionary with a valid model_version.")
    
    # If the model version is v1 and ROI cropping is enabled, raise an error since ROI cropping is not supported in v1
    if isinstance(model_config, NNInteractiveV1ModelConfig) and config.crop_roi_config.enable:
        raise ValueError("ROI cropping is not supported in NNInteractiveV1ModelConfig. Please set crop_roi_config.enable to False for model version v1.")
    
    return config, model_config
