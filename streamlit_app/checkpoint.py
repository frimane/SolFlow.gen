import streamlit as st

from solflow_gen.models import load_model

MAX_CHECKPOINT_BYTES = 512 * 1024 * 1024


def validate_generator_checkpoint(model):
    # The app only runs the flow generator, and needs the day-grid layout that
    # training records in the checkpoint.
    if getattr(model, "family", None) != "flow":
        raise ValueError(f"Only flow checkpoints are supported, got {getattr(model, 'family', None)!r}.")
    if (model.extra or {}).get("noon_col") is None:
        raise ValueError("This checkpoint does not record the day-grid layout (noon_col); retrain it.")


@st.cache_resource(show_spinner=False)
def load_checkpoint(path_text: str, modified_at: float):
    # modified_at is only part of the cache key, so a replaced file is reloaded.
    model = load_model(path_text, device="auto")
    validate_generator_checkpoint(model)
    return model
