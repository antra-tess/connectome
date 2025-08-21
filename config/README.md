# Decider Configuration

This directory contains example configuration for the split deciders:
- Pre-Decider: Interrupt classification
- Post-Decider: Activation decision

## Files
- `decider.yaml.example`: Copy to one of these locations and edit:
  1. `$CONNECTOME_CONFIG_DIR/agents/{agent_id}/decider.yaml` (highest priority)
  2. `config/decider.yaml` (fallback)

## Agent Handles (mentions)
Deciders detect mentions using the agent's name/alias and optional environment-provided handles. Set one of:
- `CONNECTOME_AGENT_HANDLES="connectome,@connectome,ConnectomeBot"`
- or `AGENT_HANDLES`, `BOT_HANDLES`

These are comma-separated values. They supplement `agent_name` and `alias` exposed by the agent/space.

## Interrupt Rules
`interrupt_rules` support:
- `interrupt_event_types`: list of event types that always interrupt
- `force_normal_event_types`: list of event types that are always normal
- `conditions`: first-match-wins conditional rules. Supported types:
  - `typing_from_last_activator` → interrupts when a `typing_notification` arrives from the same sender who caused the last activation in the same conversation
  - `field_equals` with `field` (dot-path in payload) and `value`
  - `event_type` with `value`

## Activation Rules
Defaults (without any YAML):
- `message_received` with `payload.is_dm: true` → activate + preempt
- `message_received` with a mention → activate (no preempt)
You may also use:
- `activate_event_types`: always activate on types
- `preempt_event_types`: upgrade activation priority to preempt

## Focus & HUD
- `focus_selection`: hints for focus resolution when emitting `activation_call`
- `hud`: snapshot TTL and refresh policies; snapshots are refreshed by the Heartbeat at the end of pulses and after interrupts. 