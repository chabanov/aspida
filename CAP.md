# `libaspida` — C ABI over the aspida engine

A dynamic library (`libaspida.dylib`) exposing the aspida LLM inference engine to
foreign (C / Swift / any) hosts. The native macOS UI in `reactor` links it and
calls it in-process; no server, no network.

## Build

```
make lib       # -> lib/aspida/libaspida.dylib  (install_name @rpath/libaspida.dylib)
make smoke     # build + run tools/capi_smoke.c (loads the default 9B model)
```

The smoke test / a foreign host links with:

```
cc <host.c> -Iinclude -Llib/aspida -laspida \
    -Wl,-rpath,@loader_path/lib/aspida \
    -Wl,-rpath,<adalib>        # where libgnat-15.dylib / libgnarl-15.dylib live
    -Wl,-syslibroot,$(xcrun --show-sdk-path)
```

`<adalib>` is `~/.local/share/alire/toolchains/gnat_native_15.1.2_*/lib/gcc/<triple>/15.0.1/adalib`.

## Surface — `include/aspida.h`

`aspida_init`, `aspida_load`, `aspida_unload`, `aspida_last_error`,
`aspida_detect_arch`, `aspida_arch_supported`, `aspida_discover_models`,
`aspida_chat` (streaming via `aspida_sink_t` callbacks), `aspida_vocab_size`,
`aspida_arch_name`, `aspida_free_string`.

**Memory:** `char*` returns are caller-freed via `aspida_free_string`.
`aspida_last_error` returns a pointer into a library buffer — do **not** free.
The `aspida_engine_t` is freed by `aspida_unload` (exactly once).

**Threading:** `aspida_chat` is synchronous and long; run it off the UI thread.
Every wrapper auto-registers the calling foreign thread (see below), so any
thread may call in.

## Three non-obvious gotchas (verified on macOS 27 / GNAT 15.1.2)

### 1. The library MUST be a standalone library with auto-init

`aspida_lib.gpr` sets `for Library_Interface use ("Aspida_CAPI");` and
`for Library_Auto_Init use "true";`. This makes gprbuild generate a binder
elaboration (`b__aspida.adb`) and register it as a dyld constructor.

**Why:** a plain library project links only the per-unit `___elabb/___elabs`
elaboration stubs and **never calls them**. The Ada runtime (`System.Soft_Links`,
the secondary-stack pool, the environment task) is then never initialised when
the dylib loads. The first call that returns an unconstrained `String`
(`aspida_detect_arch`, `aspida_chat`, …) dereferences a null secondary stack
and segfaults inside `system__secondary_stack__ss_mark` (`s-secsta.adb`).

With auto-init, the constructor runs the binder elaboration at load time, so
the runtime is live before any C call.

### 2. Foreign threads must be registered for a secondary stack

`Get_Sec_Stack` (used by `SS_Mark`) reads the thread's Thread-Specific Data
**directly**; it does **not** auto-register an unknown thread. A foreign thread
that never went through `System.Task_Primitives.Operations.Specific.Self` has
no TSD → null secondary stack → same `SS_Mark` segfault.

Every exported wrapper calls `Ensure_Registered`, which touches
`Ada.Task_Identification.Current_Task`. Once the runtime is elaborated (gotcha
1), that triggers `Self`, which calls `Register_Foreign_Thread` for an unknown
thread — allocating an ATCB and a secondary stack for it. It is idempotent
(`Self` checks `pthread_getspecific` first), so it is cheap on warm threads and
covers hosts that call from freshly-spawned background threads (e.g. Swift
`Task.detached`).

### 3. `C_Sink` MUST override `Emit`, not just `On_Text`

The engine streams each answer token through `Token_Sink.Emit`. The base
`Chat_Sink.Emit` (in `llm_qwen.adb`) forwards to `On_Text`, and its comment
claims "a sink that overrides only On_Text still receives text". **That is
wrong.** `Chat_Sink.Emit`'s `On_Text (S, Piece)` has `S : in out Chat_Sink`
(a *specific* tagged type). In Ada a primitive call on a specific-typed
operand is **statically bound** to that type's primitive — it does **not**
redispatch to a descendant's override (dispatching requires a class-wide
operand). So a `C_Sink` that overrides only `On_Text` never receives text:
`Sink.Emit` dispatches to the base `Chat_Sink.Emit`, whose `On_Text` call is
statically bound to the base null `On_Text` → dropped.

`On_Reasoning` / `On_Finish_Reason` / `On_Tool_Call` are fine because the engine
calls them directly on the class-wide sink (which `C_Sink` overrides), so they
dispatch correctly. Only `Emit` has the base-wrapper indirection.

**Fix:** `C_Sink` overrides `Emit` to call its own `On_Text` (with
`S : in out C_Sink`, the static binding lands on `C_Sink.On_Text`, which invokes
the C `on_text` fn pointer). This mirrors the server's `Enc_Sink`
(`src/server/encrypting_sink.adb`), which overrides `Emit` for the same reason.

Verified: a Swift host calling `aspida_chat` with prompt "What is the capital of
France?" streams `Paris` via the `on_text` callback (1 text event, finish=stop).

## macOS 27 link notes

- Compile/link with `MACOSX_DEPLOYMENT_TARGET=26.5` (the bundled GNAT was built
  for darwin23 and otherwise emits an invalid `-mmacosx-version-min`).
- The system `clang`/`gcc` default syslibroot points at a nonexistent
  `MacOSX14.sdk`; pass `-Wl,-syslibroot,$(xcrun --show-sdk-path)` explicitly
  (the Makefile `lib` target does this; gprbuild ignores `package Linker`
  switches for library projects, so the link is done manually with the gcc
  driver).
- `libaspida.dylib` depends on `@rpath/libgnat-15.dylib` and
  `@rpath/libgnarl-15.dylib`. For a self-contained `.app`, copy both into
  `Contents/Frameworks/` alongside `libaspida.dylib` and let the app's
  `@loader_path/Frameworks` rpath resolve them (their install_names are already
  `@rpath/…`). For development builds, an rpath to `<adalib>` suffices.