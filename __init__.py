from .nodes import LoadOmniGen2Image, LoadOmniGen2Model, OmniGen2, SaveOmniGen2Image

NODE_CLASS_MAPPINGS = {
    "LoadOmniGen2Image": LoadOmniGen2Image,
    "LoadOmniGen2Model": LoadOmniGen2Model,
    "OmniGen2": OmniGen2,
    "SaveOmniGen2Image": SaveOmniGen2Image,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LoadOmniGen2Image": "Load OmniGen2 Image",
    "LoadOmniGen2Model": "Load OmniGen2 Model",
    "OmniGen2": "OmniGen2",
    "SaveOmniGen2Image": "Save OmniGen2 Image",
} 

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
