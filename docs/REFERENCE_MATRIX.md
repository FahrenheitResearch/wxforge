# Reference Matrix

This file records what to learn from the existing repos without inheriting their structure.

## `metrust-py`

Reference repo: [metrust-py](C:/Users/drew/metrust-py)

Best aspect:
- scientific parity against `metpy.calc`

Fastest / strongest area:
- verified calculation workflows and parity regression corpus

Carry forward:
- formula choices
- constants
- external benchmark corpus
- parity tests and validation philosophy

Do not carry forward:
- Python-first compatibility shell as the core architecture

## `rustmet`

Reference repo: [rustmet](C:/Users/drew/rustmet)

Best aspect:
- broad operational weather platform shape

Fastest / strongest area:
- fetch + GRIB decode + serving pipeline organization

Carry forward:
- crate boundary ideas
- operational scope
- CLI/server concepts

Do not carry forward:
- duplicated foundational crates across repos
- mixed core and app concerns in one ownership model

## `rustdar`

Reference repo: [rustdar](C:/Users/drew/rustdar)

Best aspect:
- radar-specific algorithms and workstation UX ideas

Fastest / strongest area:
- Level II rendering, meso/TVS logic, color table handling, radar-centric products

Carry forward:
- detection logic
- radar rendering concepts
- color-table-as-data-transform support

Do not carry forward:
- UI-first app boundary as the reusable core

## `rusbie`

Reference repo: [rusbie](C:/Users/drew/rusbie)

Best aspect:
- Herbie-like data access ergonomics

Fastest / strongest area:
- source probing and parallel byte-range download ideas

Carry forward:
- model/source template logic
- cache and byte-range planning ideas

Do not carry forward:
- Python drop-in compatibility as the core design center

## `cfrust`

Reference repo: [cfrust](C:/Users/drew/cfrust)

Best aspect:
- GRIB/CF decoding coverage and explicit compatibility testing

Fastest / strongest area:
- pure Rust decoder and fixture-driven validation

Carry forward:
- decoder logic
- packing/grid coverage expectations
- compatibility fixtures

Do not carry forward:
- cfgrib-compatible Python API as the core boundary

## `rustplots`

Reference repo: [rustplots](C:/Users/drew/rustplots)

Best aspect:
- deterministic meteorological rendering and large visual comparison corpus

Fastest / strongest area:
- static plot generation and native render primitives

Carry forward:
- rendering primitives
- visual comparison suite
- headless output concepts

Do not carry forward:
- Python declarative plotting layer as the production core

## `rustmet-train`

Reference repo: [rustmet-train](C:/Users/drew/rustmet-train)

Best aspect:
- dataset assembly ideas and benchmark generation workflows

Fastest / strongest area:
- practical pipeline ideas for turning weather events into training records

Carry forward:
- manifest ideas
- panel/channel concepts
- study and benchmark structure

Do not carry forward:
- TorNet-driven project shape as the architecture driver
- Python-heavy orchestration as the long-term implementation
