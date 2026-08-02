from typing import Literal

type Tuple1D = tuple[int]
type Tuple2D = tuple[int, int]
type Tuple3D = tuple[int, int, int]
type TupleND = tuple[int, ...]

type Size1D = int | Tuple1D
type Size2D = int | Tuple2D
type Size3D = int | Tuple3D
type SizeND = int | TupleND

type PaddingType = Literal['valid', 'same']
type PaddingMode = Literal['zeros', 'reflect', 'replicate', 'circular']

type Padding1D = PaddingType | Size1D
type Padding2D = PaddingType | Size2D
type Padding3D = PaddingType | Size3D
type PaddingND = str | SizeND
