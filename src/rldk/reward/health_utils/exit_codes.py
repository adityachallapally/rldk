"""Compatibility shim for exit_codes imports.

This module provides backward compatibility for imports from rldk.reward.health_utils.exit_codes.
The actual implementation has been moved to rldk.reward.health_config.exit_codes.
"""

# Import from the new location
from rldk.reward.health_config.exit_codes import get_exit_code, raise_on_failure

# Re-export for backward compatibility
__all__ = ['get_exit_code', 'raise_on_failure']
