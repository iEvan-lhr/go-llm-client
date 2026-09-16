# Local client integration

Based on `v0.0.0-20260916022530-5cb8648bbdec`. The parent module uses this
source through a relative Go replace directive so the changes are reproducible.

- Preserve provider usage, including cache read/write counters, for ordinary
  and streaming compatible chat responses. Missing counters remain unknown.
- Keep independent message calls and request configuration propagation consistent.
- Apply cache defaults consistently to Responses creation with explicit request
  values taking precedence.
- Expose configurable prompt-cache capabilities for compatible endpoints.
  Nil capabilities preserve explicitly configured fields; non-nil capabilities
  independently gate key, retention, and options.
- Request usage in OpenAI Chat Completions streams by default. An explicit
  Parameters["stream_options"] value overrides that default for compatible
  gateways. Normalize compatible usage counters on this path as well.

No credentials or provider responses are stored in this directory.
