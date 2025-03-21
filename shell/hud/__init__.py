"""
HUD (Heads-Up Display)

The HUD is responsible for rendering the context presented to the agent.
It collects renderings from Element Delegates and applies transformations
like compression, summarization, and filtering.
"""

from .core import HUD, RenderingRequest, RenderingResponse

__all__ = ['HUD', 'RenderingRequest', 'RenderingResponse'] 