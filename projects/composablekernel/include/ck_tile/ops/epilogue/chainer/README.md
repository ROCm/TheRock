# CK Tile Epilogue Chainer

## Overview

The Epilogue Chainer provides a modular epilogue processing framework through scheduler-defined operation graphs.

## Architecture

### Core Design Principle
The chainer follows a **Scheduler-Graph-Node** architecture with shared context:
- **Scheduler**: Defines operation graphs and creates a shared context
- **Graph**: Composes multiple operations into sequential processing units
- **Node**: Wraps individual epilogue operations with their arguments

### EpilogueChainer
The `EpilogueChainer` struct serves as the modular epilogue processing facilitator. It delegates to schedulers for context creation and schedule generation, then processes the resulting operation graphs.

### EpilogueNode
Individual epilogue operations are wrapped in `EpilogueNode` structures that capture required arguments at construction time and automatically forward them during processing. Supports both parameterized and parameter-free operations.

### EpilogueGraph  
The `EpilogueGraph` composes multiple nodes into sequential processing units that iterate over multiple accesses if needed, running all operations in order for each iteration.

## Files

### Core Infrastructure
- `epilogue_chainer.hpp` - General chainer, node, and graph infrastructure
- `common_epilogue_ops.hpp` - Epilogue operations usable with any epilogue type

### CShuffle Implementation
- `cshuffle_epilogue_chainer_ops.hpp` - CShuffle-specific problem, context, and slice operations
- `cshuffle_epilogue_schedule.hpp` - CShuffle scheduler with pre-built schedules

## Usage

### Common Operations (common_epilogue_ops.hpp)
These operations work with any context that provides the standardized interface:
- `ScaleScalarOp` - Scale working-tile by scalar values
- `CastAndStoreToLdsOp<DstType>` - Cast working-tile and store to LDS
- `LoadFromLdsOp<Pattern>` - Load output tile from LDS with sync
- `ElementwiseOp<Func, NumAux>` - Apply elementwise operation with auxiliary tensors
- `StoreOp<MemOp>` - Store output tile to global memory
- `MoveWindowsOp<SFC, NumAux>` - Advance windows to next position

### CShuffle-Specific Operations (cshuffle_epilogue_chainer_ops.hpp)
These operations are specific to CShuffle epilogue:
- `CShuffleSliceOp` - Slice accumulator tile based on distribution
- `CShuffleScaleWindowOp` - Scale using tensor windows with shuffle distribution

### Context Interface
Operations communicate through a shared context with standardized members:
- `working_tile`: Tile for intermediate computations
- `out_tile`: Output tile
- `aux_windows`: Tuple of auxiliary tensor windows
- `lds_write_window`: Window for writing to LDS
- `lds_read_window`: Window for reading from LDS

### Schedule Tags
- `DefaultScheduleTag` - Standard: Slice → CastStore → Load → ApplyD → Store → Move
- `RowColQuantScheduleTag` - With window scaling
- `TensorQuantScheduleTag` - With scalar scaling 