ţĆ
ĺ%¸%
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	
î
	ApplyAdam
var"T	
m"T	
v"T
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
P
Assert
	condition
	
data2T"
T
list(type)(0"
	summarizeint
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
s
	AssignAdd
ref"T

value"T

output_ref"T" 
Ttype:
2	"
use_lockingbool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
š
DenseToDenseSetOperation	
set1"T	
set2"T
result_indices	
result_values"T
result_shape	"
set_operationstring"
validate_indicesbool("
Ttype:
	2	
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
B
Equal
x"T
y"T
z
"
Ttype:
2	

W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
,
Floor
x"T
y"T"
Ttype:
2
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
2
L2Loss
t"T
output"T"
Ttype:
2
,
Log
x"T
y"T"
Ttype:

2
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
;
Maximum
x"T
y"T
z"T"
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
8
MergeSummary
inputs*N
summary"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
)
Rank

input"T

output"	
Ttype
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
G
SquaredDifference
x"T
y"T
z"T"
Ttype:

2	
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.11.02v1.11.0-rc2-4-gc19e29306cÇě
n
model_inputPlaceholder*
dtype0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
shape:˙˙˙˙˙˙˙˙˙
R
model_targetsPlaceholder*
shape:*
dtype0*
_output_shapes
:
L
PlaceholderPlaceholder*
dtype0*
_output_shapes
: *
shape: 
U
model_istrainingPlaceholder*
dtype0
*
_output_shapes
:*
shape:
G
ConstConst*
value	B : *
dtype0*
_output_shapes
: 
o
global_step
VariableV2*
shape: *
shared_name *
dtype0*
	container *
_output_shapes
: 

global_step/AssignAssignglobal_stepConst*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
T0*
_class
loc:@global_step*
_output_shapes
: 
ł
7layer_fc0/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"       *)
_class
loc:@layer_fc0/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc0/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *ňę­ž*)
_class
loc:@layer_fc0/dense/kernel*
dtype0*
_output_shapes
: 
Ľ
5layer_fc0/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *ňę­>*)
_class
loc:@layer_fc0/dense/kernel*
dtype0*
_output_shapes
: 

?layer_fc0/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7layer_fc0/dense/kernel/Initializer/random_uniform/shape*)
_class
loc:@layer_fc0/dense/kernel*
seed2 *
dtype0*
_output_shapes

: *

seed *
T0
ö
5layer_fc0/dense/kernel/Initializer/random_uniform/subSub5layer_fc0/dense/kernel/Initializer/random_uniform/max5layer_fc0/dense/kernel/Initializer/random_uniform/min*)
_class
loc:@layer_fc0/dense/kernel*
_output_shapes
: *
T0

5layer_fc0/dense/kernel/Initializer/random_uniform/mulMul?layer_fc0/dense/kernel/Initializer/random_uniform/RandomUniform5layer_fc0/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@layer_fc0/dense/kernel*
_output_shapes

: 
ú
1layer_fc0/dense/kernel/Initializer/random_uniformAdd5layer_fc0/dense/kernel/Initializer/random_uniform/mul5layer_fc0/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc0/dense/kernel*
_output_shapes

: 
ľ
layer_fc0/dense/kernel
VariableV2*
dtype0*
_output_shapes

: *
shared_name *)
_class
loc:@layer_fc0/dense/kernel*
	container *
shape
: 
ď
layer_fc0/dense/kernel/AssignAssignlayer_fc0/dense/kernel1layer_fc0/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

: *
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel

layer_fc0/dense/kernel/readIdentitylayer_fc0/dense/kernel*)
_class
loc:@layer_fc0/dense/kernel*
_output_shapes

: *
T0
§
7layer_fc0/dense/kernel/Regularizer/l2_regularizer/scaleConst*
dtype0*
_output_shapes
: *
valueB
 *ŹĹ'7*)
_class
loc:@layer_fc0/dense/kernel
Ť
8layer_fc0/dense/kernel/Regularizer/l2_regularizer/L2LossL2Losslayer_fc0/dense/kernel/read*
_output_shapes
: *
T0*)
_class
loc:@layer_fc0/dense/kernel
÷
1layer_fc0/dense/kernel/Regularizer/l2_regularizerMul7layer_fc0/dense/kernel/Regularizer/l2_regularizer/scale8layer_fc0/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*)
_class
loc:@layer_fc0/dense/kernel*
_output_shapes
: 

&layer_fc0/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *'
_class
loc:@layer_fc0/dense/bias
Š
layer_fc0/dense/bias
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@layer_fc0/dense/bias*
	container 
Ú
layer_fc0/dense/bias/AssignAssignlayer_fc0/dense/bias&layer_fc0/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(*
_output_shapes
: 

layer_fc0/dense/bias/readIdentitylayer_fc0/dense/bias*
_output_shapes
: *
T0*'
_class
loc:@layer_fc0/dense/bias
˘
layer_fc0/dense/MatMulMatMulmodel_inputlayer_fc0/dense/kernel/read*
transpose_b( *
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 

layer_fc0/dense/BiasAddBiasAddlayer_fc0/dense/MatMullayer_fc0/dense/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*
data_formatNHWC
g
layer_fc0/dense/ReluRelulayer_fc0/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ł
7layer_fc1/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"    @   *)
_class
loc:@layer_fc1/dense/kernel
Ľ
5layer_fc1/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *  ž*)
_class
loc:@layer_fc1/dense/kernel*
dtype0
Ľ
5layer_fc1/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *  >*)
_class
loc:@layer_fc1/dense/kernel*
dtype0*
_output_shapes
: 

?layer_fc1/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7layer_fc1/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

: @*

seed *
T0*)
_class
loc:@layer_fc1/dense/kernel*
seed2 
ö
5layer_fc1/dense/kernel/Initializer/random_uniform/subSub5layer_fc1/dense/kernel/Initializer/random_uniform/max5layer_fc1/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc1/dense/kernel*
_output_shapes
: 

5layer_fc1/dense/kernel/Initializer/random_uniform/mulMul?layer_fc1/dense/kernel/Initializer/random_uniform/RandomUniform5layer_fc1/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@layer_fc1/dense/kernel*
_output_shapes

: @
ú
1layer_fc1/dense/kernel/Initializer/random_uniformAdd5layer_fc1/dense/kernel/Initializer/random_uniform/mul5layer_fc1/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc1/dense/kernel*
_output_shapes

: @
ľ
layer_fc1/dense/kernel
VariableV2*
dtype0*
_output_shapes

: @*
shared_name *)
_class
loc:@layer_fc1/dense/kernel*
	container *
shape
: @
ď
layer_fc1/dense/kernel/AssignAssignlayer_fc1/dense/kernel1layer_fc1/dense/kernel/Initializer/random_uniform*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @*
use_locking(

layer_fc1/dense/kernel/readIdentitylayer_fc1/dense/kernel*
_output_shapes

: @*
T0*)
_class
loc:@layer_fc1/dense/kernel
§
7layer_fc1/dense/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'7*)
_class
loc:@layer_fc1/dense/kernel*
dtype0*
_output_shapes
: 
Ť
8layer_fc1/dense/kernel/Regularizer/l2_regularizer/L2LossL2Losslayer_fc1/dense/kernel/read*
_output_shapes
: *
T0*)
_class
loc:@layer_fc1/dense/kernel
÷
1layer_fc1/dense/kernel/Regularizer/l2_regularizerMul7layer_fc1/dense/kernel/Regularizer/l2_regularizer/scale8layer_fc1/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*)
_class
loc:@layer_fc1/dense/kernel*
_output_shapes
: 

&layer_fc1/dense/bias/Initializer/zerosConst*
valueB@*    *'
_class
loc:@layer_fc1/dense/bias*
dtype0*
_output_shapes
:@
Š
layer_fc1/dense/bias
VariableV2*
shared_name *'
_class
loc:@layer_fc1/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
Ú
layer_fc1/dense/bias/AssignAssignlayer_fc1/dense/bias&layer_fc1/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@

layer_fc1/dense/bias/readIdentitylayer_fc1/dense/bias*
T0*'
_class
loc:@layer_fc1/dense/bias*
_output_shapes
:@
Ť
layer_fc1/dense/MatMulMatMullayer_fc0/dense/Relulayer_fc1/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
transpose_b( 

layer_fc1/dense/BiasAddBiasAddlayer_fc1/dense/MatMullayer_fc1/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
g
layer_fc1/dense/ReluRelulayer_fc1/dense/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ł
7layer_fc2/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *)
_class
loc:@layer_fc2/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc2/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *ó5ž*)
_class
loc:@layer_fc2/dense/kernel
Ľ
5layer_fc2/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *ó5>*)
_class
loc:@layer_fc2/dense/kernel

?layer_fc2/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7layer_fc2/dense/kernel/Initializer/random_uniform/shape*)
_class
loc:@layer_fc2/dense/kernel*
seed2 *
dtype0*
_output_shapes
:	@*

seed *
T0
ö
5layer_fc2/dense/kernel/Initializer/random_uniform/subSub5layer_fc2/dense/kernel/Initializer/random_uniform/max5layer_fc2/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@layer_fc2/dense/kernel

5layer_fc2/dense/kernel/Initializer/random_uniform/mulMul?layer_fc2/dense/kernel/Initializer/random_uniform/RandomUniform5layer_fc2/dense/kernel/Initializer/random_uniform/sub*
_output_shapes
:	@*
T0*)
_class
loc:@layer_fc2/dense/kernel
ű
1layer_fc2/dense/kernel/Initializer/random_uniformAdd5layer_fc2/dense/kernel/Initializer/random_uniform/mul5layer_fc2/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
:	@
ˇ
layer_fc2/dense/kernel
VariableV2*
shape:	@*
dtype0*
_output_shapes
:	@*
shared_name *)
_class
loc:@layer_fc2/dense/kernel*
	container 
đ
layer_fc2/dense/kernel/AssignAssignlayer_fc2/dense/kernel1layer_fc2/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@

layer_fc2/dense/kernel/readIdentitylayer_fc2/dense/kernel*
T0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
:	@
§
7layer_fc2/dense/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'7*)
_class
loc:@layer_fc2/dense/kernel*
dtype0*
_output_shapes
: 
Ť
8layer_fc2/dense/kernel/Regularizer/l2_regularizer/L2LossL2Losslayer_fc2/dense/kernel/read*
T0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
: 
÷
1layer_fc2/dense/kernel/Regularizer/l2_regularizerMul7layer_fc2/dense/kernel/Regularizer/l2_regularizer/scale8layer_fc2/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
: 

&layer_fc2/dense/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc2/dense/bias*
dtype0*
_output_shapes	
:
Ť
layer_fc2/dense/bias
VariableV2*'
_class
loc:@layer_fc2/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
Ű
layer_fc2/dense/bias/AssignAssignlayer_fc2/dense/bias&layer_fc2/dense/bias/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias

layer_fc2/dense/bias/readIdentitylayer_fc2/dense/bias*
_output_shapes	
:*
T0*'
_class
loc:@layer_fc2/dense/bias
Ź
layer_fc2/dense/MatMulMatMullayer_fc1/dense/Relulayer_fc2/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0

layer_fc2/dense/BiasAddBiasAddlayer_fc2/dense/MatMullayer_fc2/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
layer_fc2/dense/ReluRelulayer_fc2/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7layer_fc3/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@layer_fc3/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc3/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*)
_class
loc:@layer_fc3/dense/kernel*
dtype0*
_output_shapes
: 
Ľ
5layer_fc3/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*)
_class
loc:@layer_fc3/dense/kernel*
dtype0*
_output_shapes
: 

?layer_fc3/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7layer_fc3/dense/kernel/Initializer/random_uniform/shape*
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@layer_fc3/dense/kernel*
seed2 
ö
5layer_fc3/dense/kernel/Initializer/random_uniform/subSub5layer_fc3/dense/kernel/Initializer/random_uniform/max5layer_fc3/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc3/dense/kernel*
_output_shapes
: 

5layer_fc3/dense/kernel/Initializer/random_uniform/mulMul?layer_fc3/dense/kernel/Initializer/random_uniform/RandomUniform5layer_fc3/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@layer_fc3/dense/kernel* 
_output_shapes
:

ü
1layer_fc3/dense/kernel/Initializer/random_uniformAdd5layer_fc3/dense/kernel/Initializer/random_uniform/mul5layer_fc3/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc3/dense/kernel* 
_output_shapes
:

š
layer_fc3/dense/kernel
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@layer_fc3/dense/kernel*
	container *
shape:

ń
layer_fc3/dense/kernel/AssignAssignlayer_fc3/dense/kernel1layer_fc3/dense/kernel/Initializer/random_uniform* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(

layer_fc3/dense/kernel/readIdentitylayer_fc3/dense/kernel* 
_output_shapes
:
*
T0*)
_class
loc:@layer_fc3/dense/kernel
§
7layer_fc3/dense/kernel/Regularizer/l2_regularizer/scaleConst*
_output_shapes
: *
valueB
 *ŹĹ'7*)
_class
loc:@layer_fc3/dense/kernel*
dtype0
Ť
8layer_fc3/dense/kernel/Regularizer/l2_regularizer/L2LossL2Losslayer_fc3/dense/kernel/read*)
_class
loc:@layer_fc3/dense/kernel*
_output_shapes
: *
T0
÷
1layer_fc3/dense/kernel/Regularizer/l2_regularizerMul7layer_fc3/dense/kernel/Regularizer/l2_regularizer/scale8layer_fc3/dense/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*)
_class
loc:@layer_fc3/dense/kernel

&layer_fc3/dense/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc3/dense/bias*
dtype0*
_output_shapes	
:
Ť
layer_fc3/dense/bias
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@layer_fc3/dense/bias*
	container *
shape:
Ű
layer_fc3/dense/bias/AssignAssignlayer_fc3/dense/bias&layer_fc3/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(*
_output_shapes	
:

layer_fc3/dense/bias/readIdentitylayer_fc3/dense/bias*'
_class
loc:@layer_fc3/dense/bias*
_output_shapes	
:*
T0
Ź
layer_fc3/dense/MatMulMatMullayer_fc2/dense/Relulayer_fc3/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0

layer_fc3/dense/BiasAddBiasAddlayer_fc3/dense/MatMullayer_fc3/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
layer_fc3/dense/ReluRelulayer_fc3/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7layer_fc4/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc4/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *qÄž*)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
: 
Ľ
5layer_fc4/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *qÄ>*)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
: 

?layer_fc4/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7layer_fc4/dense/kernel/Initializer/random_uniform/shape*
seed2 *
dtype0* 
_output_shapes
:
*

seed *
T0*)
_class
loc:@layer_fc4/dense/kernel
ö
5layer_fc4/dense/kernel/Initializer/random_uniform/subSub5layer_fc4/dense/kernel/Initializer/random_uniform/max5layer_fc4/dense/kernel/Initializer/random_uniform/min*)
_class
loc:@layer_fc4/dense/kernel*
_output_shapes
: *
T0

5layer_fc4/dense/kernel/Initializer/random_uniform/mulMul?layer_fc4/dense/kernel/Initializer/random_uniform/RandomUniform5layer_fc4/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@layer_fc4/dense/kernel* 
_output_shapes
:

ü
1layer_fc4/dense/kernel/Initializer/random_uniformAdd5layer_fc4/dense/kernel/Initializer/random_uniform/mul5layer_fc4/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc4/dense/kernel* 
_output_shapes
:

š
layer_fc4/dense/kernel
VariableV2*
shared_name *)
_class
loc:@layer_fc4/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ń
layer_fc4/dense/kernel/AssignAssignlayer_fc4/dense/kernel1layer_fc4/dense/kernel/Initializer/random_uniform*
T0*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(

layer_fc4/dense/kernel/readIdentitylayer_fc4/dense/kernel*)
_class
loc:@layer_fc4/dense/kernel* 
_output_shapes
:
*
T0
§
7layer_fc4/dense/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'7*)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
: 
Ť
8layer_fc4/dense/kernel/Regularizer/l2_regularizer/L2LossL2Losslayer_fc4/dense/kernel/read*
_output_shapes
: *
T0*)
_class
loc:@layer_fc4/dense/kernel
÷
1layer_fc4/dense/kernel/Regularizer/l2_regularizerMul7layer_fc4/dense/kernel/Regularizer/l2_regularizer/scale8layer_fc4/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*)
_class
loc:@layer_fc4/dense/kernel*
_output_shapes
: 

&layer_fc4/dense/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc4/dense/bias*
dtype0*
_output_shapes	
:
Ť
layer_fc4/dense/bias
VariableV2*
shared_name *'
_class
loc:@layer_fc4/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
Ű
layer_fc4/dense/bias/AssignAssignlayer_fc4/dense/bias&layer_fc4/dense/bias/Initializer/zeros*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

layer_fc4/dense/bias/readIdentitylayer_fc4/dense/bias*
T0*'
_class
loc:@layer_fc4/dense/bias*
_output_shapes	
:
Ź
layer_fc4/dense/MatMulMatMullayer_fc3/dense/Relulayer_fc4/dense/kernel/read*
T0*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( 

layer_fc4/dense/BiasAddBiasAddlayer_fc4/dense/MatMullayer_fc4/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
layer_fc4/dense/ReluRelulayer_fc4/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ł
7layer_fc5/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *)
_class
loc:@layer_fc5/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc5/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *qÄž*)
_class
loc:@layer_fc5/dense/kernel
Ľ
5layer_fc5/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *qÄ>*)
_class
loc:@layer_fc5/dense/kernel

?layer_fc5/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7layer_fc5/dense/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@layer_fc5/dense/kernel*
seed2 *
dtype0* 
_output_shapes
:
*

seed 
ö
5layer_fc5/dense/kernel/Initializer/random_uniform/subSub5layer_fc5/dense/kernel/Initializer/random_uniform/max5layer_fc5/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc5/dense/kernel*
_output_shapes
: 

5layer_fc5/dense/kernel/Initializer/random_uniform/mulMul?layer_fc5/dense/kernel/Initializer/random_uniform/RandomUniform5layer_fc5/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@layer_fc5/dense/kernel* 
_output_shapes
:

ü
1layer_fc5/dense/kernel/Initializer/random_uniformAdd5layer_fc5/dense/kernel/Initializer/random_uniform/mul5layer_fc5/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@layer_fc5/dense/kernel* 
_output_shapes
:

š
layer_fc5/dense/kernel
VariableV2*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@layer_fc5/dense/kernel
ń
layer_fc5/dense/kernel/AssignAssignlayer_fc5/dense/kernel1layer_fc5/dense/kernel/Initializer/random_uniform*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0

layer_fc5/dense/kernel/readIdentitylayer_fc5/dense/kernel*
T0*)
_class
loc:@layer_fc5/dense/kernel* 
_output_shapes
:

§
7layer_fc5/dense/kernel/Regularizer/l2_regularizer/scaleConst*
valueB
 *ŹĹ'7*)
_class
loc:@layer_fc5/dense/kernel*
dtype0*
_output_shapes
: 
Ť
8layer_fc5/dense/kernel/Regularizer/l2_regularizer/L2LossL2Losslayer_fc5/dense/kernel/read*
_output_shapes
: *
T0*)
_class
loc:@layer_fc5/dense/kernel
÷
1layer_fc5/dense/kernel/Regularizer/l2_regularizerMul7layer_fc5/dense/kernel/Regularizer/l2_regularizer/scale8layer_fc5/dense/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0*)
_class
loc:@layer_fc5/dense/kernel

&layer_fc5/dense/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc5/dense/bias*
dtype0*
_output_shapes	
:
Ť
layer_fc5/dense/bias
VariableV2*
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@layer_fc5/dense/bias*
	container 
Ű
layer_fc5/dense/bias/AssignAssignlayer_fc5/dense/bias&layer_fc5/dense/bias/Initializer/zeros*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(

layer_fc5/dense/bias/readIdentitylayer_fc5/dense/bias*'
_class
loc:@layer_fc5/dense/bias*
_output_shapes	
:*
T0
Ź
layer_fc5/dense/MatMulMatMullayer_fc4/dense/Relulayer_fc5/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0

layer_fc5/dense/BiasAddBiasAddlayer_fc5/dense/MatMullayer_fc5/dense/bias/read*
T0*
data_formatNHWC*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
layer_fc5/dense/ReluRelulayer_fc5/dense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
dropout/cond/SwitchSwitchmodel_istrainingmodel_istraining*
_output_shapes

::*
T0

[
dropout/cond/switch_tIdentitydropout/cond/Switch:1*
T0
*
_output_shapes
:
Y
dropout/cond/switch_fIdentitydropout/cond/Switch*
T0
*
_output_shapes
:
U
dropout/cond/pred_idIdentitymodel_istraining*
_output_shapes
:*
T0

{
dropout/cond/dropout/keep_probConst^dropout/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *ÍĚL?
}
dropout/cond/dropout/ShapeShape#dropout/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
Ç
!dropout/cond/dropout/Shape/SwitchSwitchlayer_fc5/dense/Reludropout/cond/pred_id*
T0*'
_class
loc:@layer_fc5/dense/Relu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

'dropout/cond/dropout/random_uniform/minConst^dropout/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    

'dropout/cond/dropout/random_uniform/maxConst^dropout/cond/switch_t*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ˇ
1dropout/cond/dropout/random_uniform/RandomUniformRandomUniformdropout/cond/dropout/Shape*
seed2 *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*

seed *
T0*
dtype0
Ą
'dropout/cond/dropout/random_uniform/subSub'dropout/cond/dropout/random_uniform/max'dropout/cond/dropout/random_uniform/min*
_output_shapes
: *
T0
˝
'dropout/cond/dropout/random_uniform/mulMul1dropout/cond/dropout/random_uniform/RandomUniform'dropout/cond/dropout/random_uniform/sub*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ż
#dropout/cond/dropout/random_uniformAdd'dropout/cond/dropout/random_uniform/mul'dropout/cond/dropout/random_uniform/min*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout/cond/dropout/addAdddropout/cond/dropout/keep_prob#dropout/cond/dropout/random_uniform*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
p
dropout/cond/dropout/FloorFloordropout/cond/dropout/add*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout/cond/dropout/divRealDiv#dropout/cond/dropout/Shape/Switch:1dropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dropout/cond/dropout/mulMuldropout/cond/dropout/divdropout/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
r
dropout/cond/IdentityIdentitydropout/cond/Identity/Switch*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
dropout/cond/Identity/SwitchSwitchlayer_fc5/dense/Reludropout/cond/pred_id*
T0*'
_class
loc:@layer_fc5/dense/Relu*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

dropout/cond/MergeMergedropout/cond/Identitydropout/cond/dropout/mul*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: *
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
valueB"      *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
valueB
 *n×\ž*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *n×\>*
_class
loc:@dense/kernel*
dtype0
ć
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	*

seed *
T0*
_class
loc:@dense/kernel*
seed2 
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
_output_shapes
: *
T0
á
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	
Ó
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	*
T0*
_class
loc:@dense/kernel
Ł
dense/kernel
VariableV2*
_output_shapes
:	*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	*
dtype0
Č
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	
v
dense/kernel/readIdentitydense/kernel*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	

dense/MatMulMatMuldropout/cond/Mergedense/kernel/read*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b( *
T0
R

dense/ReluReludense/MatMul*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
i
model_prediction/shapeConst*
valueB:
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
:
{
model_predictionReshape
dense/Relumodel_prediction/shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ý
total_regularization_lossAddN1layer_fc0/dense/kernel/Regularizer/l2_regularizer1layer_fc1/dense/kernel/Regularizer/l2_regularizer1layer_fc2/dense/kernel/Regularizer/l2_regularizer1layer_fc3/dense/kernel/Regularizer/l2_regularizer1layer_fc4/dense/kernel/Regularizer/l2_regularizer1layer_fc5/dense/kernel/Regularizer/l2_regularizer*
T0*
N*
_output_shapes
: 
J
add/xConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
C
addAddadd/xmodel_targets*
_output_shapes
:*
T0
2
LogLogadd*
T0*
_output_shapes
:
L
add_1/xConst*
valueB
 *  ?*
dtype0*
_output_shapes
: 
U
add_1Addadd_1/xmodel_prediction*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
A
Log_1Logadd_1*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
h
$mean_squared_error/SquaredDifferenceSquaredDifferenceLog_1Log*
T0*
_output_shapes
:
t
/mean_squared_error/assert_broadcastable/weightsConst*
_output_shapes
: *
valueB
 *  ?*
dtype0
x
5mean_squared_error/assert_broadcastable/weights/shapeConst*
valueB *
dtype0*
_output_shapes
: 
v
4mean_squared_error/assert_broadcastable/weights/rankConst*
value	B : *
dtype0*
_output_shapes
: 
Ą
4mean_squared_error/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
out_type0

3mean_squared_error/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference*
_output_shapes
: *
T0
u
3mean_squared_error/assert_broadcastable/is_scalar/xConst*
value	B : *
dtype0*
_output_shapes
: 
Ć
1mean_squared_error/assert_broadcastable/is_scalarEqual3mean_squared_error/assert_broadcastable/is_scalar/x4mean_squared_error/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
Đ
=mean_squared_error/assert_broadcastable/is_valid_shape/SwitchSwitch1mean_squared_error/assert_broadcastable/is_scalar1mean_squared_error/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

­
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_tIdentity?mean_squared_error/assert_broadcastable/is_valid_shape/Switch:1*
T0
*
_output_shapes
: 
Ť
?mean_squared_error/assert_broadcastable/is_valid_shape/switch_fIdentity=mean_squared_error/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 

>mean_squared_error/assert_broadcastable/is_valid_shape/pred_idIdentity1mean_squared_error/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 
Ľ
?mean_squared_error/assert_broadcastable/is_valid_shape/Switch_1Switch1mean_squared_error/assert_broadcastable/is_scalar>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0
*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
_output_shapes
: : 
Ő
]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualdmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchfmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
T0*
_output_shapes
: 
Î
dmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitch3mean_squared_error/assert_broadcastable/values/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*F
_class<
:8loc:@mean_squared_error/assert_broadcastable/values/rank*
_output_shapes
: : 
Ň
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1Switch4mean_squared_error/assert_broadcastable/weights/rank>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/weights/rank*
_output_shapes
: : 
Â
Wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0

á
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentityYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
_output_shapes
: *
T0

ß
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityWmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
T0
*
_output_shapes
: 
ä
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 

pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
Ł
lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimswmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tdim0*
T0
ů
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitch4mean_squared_error/assert_broadcastable/values/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ô
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchsmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapelmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
T0*
out_type0*
_output_shapes
:

qmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 

kmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapeqmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*
T0*

index_type0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 

hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2lmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimskmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likemmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*

Tidx0*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*
_output_shapes

: *

Tdim0*
T0
ç
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitch5mean_squared_error/assert_broadcastable/weights/shape>mean_squared_error/assert_broadcastable/is_valid_shape/pred_id*
T0*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
_output_shapes

: : 
Ă
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchumean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape*
_output_shapes

: : 
ç
zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationnmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
set_operationa-b
Š
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSize|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0

cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConstZ^mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
ä
amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualcmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xrmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
ą
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch]mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankXmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0
*p
_classf
dbloc:@mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : 
É
Vmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergeYmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1amean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
_output_shapes
: : *
T0
*
N

<mean_squared_error/assert_broadcastable/is_valid_shape/MergeMergeVmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeAmean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 

-mean_squared_error/assert_broadcastable/ConstConst*
_output_shapes
: *8
value/B- B'weights can not be broadcast to values.*
dtype0
~
/mean_squared_error/assert_broadcastable/Const_1Const*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
Ą
/mean_squared_error/assert_broadcastable/Const_2Const*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
dtype0*
_output_shapes
: 
}
/mean_squared_error/assert_broadcastable/Const_3Const*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 

/mean_squared_error/assert_broadcastable/Const_4Const*7
value.B, B&mean_squared_error/SquaredDifference:0*
dtype0*
_output_shapes
: 
z
/mean_squared_error/assert_broadcastable/Const_5Const*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 
ă
:mean_squared_error/assert_broadcastable/AssertGuard/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: : 
§
<mean_squared_error/assert_broadcastable/AssertGuard/switch_tIdentity<mean_squared_error/assert_broadcastable/AssertGuard/Switch:1*
_output_shapes
: *
T0

Ľ
<mean_squared_error/assert_broadcastable/AssertGuard/switch_fIdentity:mean_squared_error/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
Ś
;mean_squared_error/assert_broadcastable/AssertGuard/pred_idIdentity<mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 

8mean_squared_error/assert_broadcastable/AssertGuard/NoOpNoOp=^mean_squared_error/assert_broadcastable/AssertGuard/switch_t
˝
Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependencyIdentity<mean_squared_error/assert_broadcastable/AssertGuard/switch_t9^mean_squared_error/assert_broadcastable/AssertGuard/NoOp*
_output_shapes
: *
T0
*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_t
č
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Ď
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
ň
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*B
value9B7 B1mean_squared_error/assert_broadcastable/weights:0*
dtype0*
_output_shapes
: 
Î
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
ç
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*7
value.B, B&mean_squared_error/SquaredDifference:0*
dtype0*
_output_shapes
: 
Ë
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Const=^mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
valueB B
is_scalar=*
dtype0*
_output_shapes
: 

:mean_squared_error/assert_broadcastable/AssertGuard/AssertAssertAmean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchAmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Amean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
ş
Amean_squared_error/assert_broadcastable/AssertGuard/Assert/SwitchSwitch<mean_squared_error/assert_broadcastable/is_valid_shape/Merge;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*
T0
*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : 
˛
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1Switch5mean_squared_error/assert_broadcastable/weights/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*
_output_shapes

: : *
T0*H
_class>
<:loc:@mean_squared_error/assert_broadcastable/weights/shape
Ć
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2Switch4mean_squared_error/assert_broadcastable/values/shape;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*G
_class=
;9loc:@mean_squared_error/assert_broadcastable/values/shape
Ś
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3Switch1mean_squared_error/assert_broadcastable/is_scalar;mean_squared_error/assert_broadcastable/AssertGuard/pred_id*
T0
*D
_class:
86loc:@mean_squared_error/assert_broadcastable/is_scalar*
_output_shapes
: : 
Á
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Identity<mean_squared_error/assert_broadcastable/AssertGuard/switch_f;^mean_squared_error/assert_broadcastable/AssertGuard/Assert*
T0
*O
_classE
CAloc:@mean_squared_error/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 

9mean_squared_error/assert_broadcastable/AssertGuard/MergeMergeHmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1Fmean_squared_error/assert_broadcastable/AssertGuard/control_dependency*
T0
*
N*
_output_shapes
: : 

mean_squared_error/ToFloat/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ?*
dtype0*
_output_shapes
: 

mean_squared_error/MulMul$mean_squared_error/SquaredDifferencemean_squared_error/ToFloat/x*
_output_shapes
:*
T0
X
mean_squared_error/RankRankmean_squared_error/Mul*
T0*
_output_shapes
: 

mean_squared_error/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
_output_shapes
: *
value	B : *
dtype0

mean_squared_error/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
value	B :*
dtype0*
_output_shapes
: 
Ť
mean_squared_error/rangeRangemean_squared_error/range/startmean_squared_error/Rankmean_squared_error/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0

mean_squared_error/SumSummean_squared_error/Mulmean_squared_error/range*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
§
&mean_squared_error/num_present/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 

$mean_squared_error/num_present/EqualEqualmean_squared_error/ToFloat/x&mean_squared_error/num_present/Equal/y*
_output_shapes
: *
T0
Ş
)mean_squared_error/num_present/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 
­
.mean_squared_error/num_present/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB *
dtype0*
_output_shapes
: 
Ż
.mean_squared_error/num_present/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ?*
dtype0*
_output_shapes
: 
Ă
(mean_squared_error/num_present/ones_likeFill.mean_squared_error/num_present/ones_like/Shape.mean_squared_error/num_present/ones_like/Const*
T0*

index_type0*
_output_shapes
: 
Ë
%mean_squared_error/num_present/SelectSelect$mean_squared_error/num_present/Equal)mean_squared_error/num_present/zeros_like(mean_squared_error/num_present/ones_like*
T0*
_output_shapes
: 
Ň
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB *
dtype0*
_output_shapes
: 
Đ
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rankConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
value	B : *
dtype0*
_output_shapes
: 
ű
Rmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ü
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rankRank$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
T0*
_output_shapes
: 
Ď
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
value	B : *
dtype0*
_output_shapes
: 
 
Omean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarEqualQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar/xRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
T0*
_output_shapes
: 
Ş
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/SwitchSwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : *
T0

é
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_tIdentity]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch:1*
_output_shapes
: *
T0

ç
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_fIdentity[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch*
T0
*
_output_shapes
: 
Ú
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_idIdentityOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
T0
*
_output_shapes
: 

]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0
*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar
ą
{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankEqualmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switchmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1*
_output_shapes
: *
T0
Ç
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/SwitchSwitchQmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
_output_shapes
: : *
T0*d
_classZ
XVloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank
Ë
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank*
_output_shapes
: : 

umean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/SwitchSwitch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
_output_shapes
: : *
T0


wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_tIdentitywmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1*
T0
*
_output_shapes
: 

wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_fIdentityumean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch*
_output_shapes
: *
T0

 
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_idIdentity{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank*
T0
*
_output_shapes
: 

mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 

mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims
ExpandDimsmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
ň
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/SwitchSwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Î
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1Switchmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape
Ű
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ShapeShapemean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims*
_output_shapes
:*
T0*
out_type0

mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 
ú
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likeFillmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shapemean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0

mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axisConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B :*
dtype0*
_output_shapes
: 

mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concatConcatV2mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDimsmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_likemean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis*
T0*
N*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0

mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dimConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
valueB :
˙˙˙˙˙˙˙˙˙*
dtype0*
_output_shapes
: 
ű
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1
ExpandDimsmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim*

Tdim0*
T0*
_output_shapes

: 
ŕ
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/SwitchSwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id*
T0*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
_output_shapes

: : 
˝
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1Switchmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switchvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
T0*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
_output_shapes

: : 
Ä
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperationDenseToDenseSetOperationmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat*
T0*
validate_indices(*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙:*
set_operationa-b
ç
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dimsSizemean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1*
_output_shapes
: *
T0*
out_type0
ú
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xConst:^mean_squared_error/assert_broadcastable/AssertGuard/Mergex^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t*
value	B : *
dtype0*
_output_shapes
: 
Ŕ
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dimsEqualmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims*
T0*
_output_shapes
: 
­
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1Switch{mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rankvmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id*
_output_shapes
: : *
T0
*
_class
loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank
Ł
tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/MergeMergewmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims*
T0
*
N*
_output_shapes
: : 
ć
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeMergetmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1*
T0
*
N*
_output_shapes
: : 
ď
Kmean_squared_error/num_present/broadcast_weights/assert_broadcastable/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Ř
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
ń
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*8
value/B- B'mean_squared_error/num_present/Select:0*
dtype0*
_output_shapes
: 
×
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_3Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
đ
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*7
value.B, B&mean_squared_error/SquaredDifference:0*
dtype0*
_output_shapes
: 
Ô
Mmean_squared_error/num_present/broadcast_weights/assert_broadcastable/Const_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB B
is_scalar=
˝
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
_output_shapes
: : *
T0

ă
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch:1*
T0
*
_output_shapes
: 
á
Zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fIdentityXmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Switch*
T0
*
_output_shapes
: 
â
Ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_idIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge*
T0
*
_output_shapes
: 
÷
Vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOpNoOp:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t
ľ
dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependencyIdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_tW^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/NoOp*
T0
*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t*
_output_shapes
: 
ŕ
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'weights can not be broadcast to values.*
dtype0*
_output_shapes
: 
Ç
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bweights.shape=*
dtype0*
_output_shapes
: 
ŕ
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*8
value/B- B'mean_squared_error/num_present/Select:0*
dtype0*
_output_shapes
: 
Ć
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
valueB Bvalues.shape=*
dtype0*
_output_shapes
: 
ß
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*7
value.B, B&mean_squared_error/SquaredDifference:0*
dtype0*
_output_shapes
: 
Ă
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7Const:^mean_squared_error/assert_broadcastable/AssertGuard/Merge[^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: *
valueB B
is_scalar=*
dtype0
×
Xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/AssertAssert_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3*
T
2	
*
	summarize
˛
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/SwitchSwitchZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/MergeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
_output_shapes
: : *
T0
*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge
Ş
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1SwitchSmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*f
_class\
ZXloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape*
_output_shapes

: : 
ž
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2SwitchRmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shapeYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0*e
_class[
YWloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3SwitchOmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalarYmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id*
T0
*b
_classX
VTloc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar*
_output_shapes
: : 
š
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1IdentityZmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_fY^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert*
T0
*m
_classc
a_loc:@mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f*
_output_shapes
: 
Ú
Wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/MergeMergefmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1dmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency*
N*
_output_shapes
: : *
T0

Ă
@mean_squared_error/num_present/broadcast_weights/ones_like/ShapeShape$mean_squared_error/SquaredDifference:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
out_type0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

@mean_squared_error/num_present/broadcast_weights/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/MergeX^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ?*
dtype0*
_output_shapes
: 
ű
:mean_squared_error/num_present/broadcast_weights/ones_likeFill@mean_squared_error/num_present/broadcast_weights/ones_like/Shape@mean_squared_error/num_present/broadcast_weights/ones_like/Const*
T0*

index_type0*
_output_shapes
:
˝
0mean_squared_error/num_present/broadcast_weightsMul%mean_squared_error/num_present/Select:mean_squared_error/num_present/broadcast_weights/ones_like*
_output_shapes
:*
T0
~
#mean_squared_error/num_present/RankRank0mean_squared_error/num_present/broadcast_weights*
_output_shapes
: *
T0
¨
*mean_squared_error/num_present/range/startConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
value	B : *
dtype0*
_output_shapes
: 
¨
*mean_squared_error/num_present/range/deltaConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
value	B :*
dtype0*
_output_shapes
: 
Ű
$mean_squared_error/num_present/rangeRange*mean_squared_error/num_present/range/start#mean_squared_error/num_present/Rank*mean_squared_error/num_present/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0
ť
mean_squared_error/num_presentSum0mean_squared_error/num_present/broadcast_weights$mean_squared_error/num_present/range*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 

mean_squared_error/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB *
dtype0*
_output_shapes
: 

mean_squared_error/Sum_1Summean_squared_error/Summean_squared_error/Const*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 

mean_squared_error/Greater/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB
 *    *
dtype0*
_output_shapes
: 

mean_squared_error/GreaterGreatermean_squared_error/num_presentmean_squared_error/Greater/y*
_output_shapes
: *
T0

mean_squared_error/Equal/yConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *    
~
mean_squared_error/EqualEqualmean_squared_error/num_presentmean_squared_error/Equal/y*
T0*
_output_shapes
: 
Ą
"mean_squared_error/ones_like/ShapeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB *
dtype0*
_output_shapes
: 
Ł
"mean_squared_error/ones_like/ConstConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
valueB
 *  ?*
dtype0*
_output_shapes
: 

mean_squared_error/ones_likeFill"mean_squared_error/ones_like/Shape"mean_squared_error/ones_like/Const*
_output_shapes
: *
T0*

index_type0

mean_squared_error/SelectSelectmean_squared_error/Equalmean_squared_error/ones_likemean_squared_error/num_present*
T0*
_output_shapes
: 
w
mean_squared_error/divRealDivmean_squared_error/Sum_1mean_squared_error/Select*
_output_shapes
: *
T0

mean_squared_error/zeros_likeConst:^mean_squared_error/assert_broadcastable/AssertGuard/Merge*
dtype0*
_output_shapes
: *
valueB
 *    

mean_squared_error/valueSelectmean_squared_error/Greatermean_squared_error/divmean_squared_error/zeros_like*
T0*
_output_shapes
: 
b
add_2Addtotal_regularization_lossmean_squared_error/value*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  ?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
>
%gradients/add_2_grad/tuple/group_depsNoOp^gradients/Fill
ľ
-gradients/add_2_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/add_2_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
ˇ
/gradients/add_2_grad/tuple/control_dependency_1Identitygradients/Fill&^gradients/add_2_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
q
9gradients/total_regularization_loss_grad/tuple/group_depsNoOp.^gradients/add_2_grad/tuple/control_dependency
ü
Agradients/total_regularization_loss_grad/tuple/control_dependencyIdentity-gradients/add_2_grad/tuple/control_dependency:^gradients/total_regularization_loss_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
ţ
Cgradients/total_regularization_loss_grad/tuple/control_dependency_1Identity-gradients/add_2_grad/tuple/control_dependency:^gradients/total_regularization_loss_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
ţ
Cgradients/total_regularization_loss_grad/tuple/control_dependency_2Identity-gradients/add_2_grad/tuple/control_dependency:^gradients/total_regularization_loss_grad/tuple/group_deps*!
_class
loc:@gradients/Fill*
_output_shapes
: *
T0
ţ
Cgradients/total_regularization_loss_grad/tuple/control_dependency_3Identity-gradients/add_2_grad/tuple/control_dependency:^gradients/total_regularization_loss_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
ţ
Cgradients/total_regularization_loss_grad/tuple/control_dependency_4Identity-gradients/add_2_grad/tuple/control_dependency:^gradients/total_regularization_loss_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
ţ
Cgradients/total_regularization_loss_grad/tuple/control_dependency_5Identity-gradients/add_2_grad/tuple/control_dependency:^gradients/total_regularization_loss_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
w
2gradients/mean_squared_error/value_grad/zeros_likeConst*
valueB
 *    *
dtype0*
_output_shapes
: 
Ú
.gradients/mean_squared_error/value_grad/SelectSelectmean_squared_error/Greater/gradients/add_2_grad/tuple/control_dependency_12gradients/mean_squared_error/value_grad/zeros_like*
T0*
_output_shapes
: 
Ü
0gradients/mean_squared_error/value_grad/Select_1Selectmean_squared_error/Greater2gradients/mean_squared_error/value_grad/zeros_like/gradients/add_2_grad/tuple/control_dependency_1*
T0*
_output_shapes
: 
¤
8gradients/mean_squared_error/value_grad/tuple/group_depsNoOp/^gradients/mean_squared_error/value_grad/Select1^gradients/mean_squared_error/value_grad/Select_1

@gradients/mean_squared_error/value_grad/tuple/control_dependencyIdentity.gradients/mean_squared_error/value_grad/Select9^gradients/mean_squared_error/value_grad/tuple/group_deps*
_output_shapes
: *
T0*A
_class7
53loc:@gradients/mean_squared_error/value_grad/Select
Ą
Bgradients/mean_squared_error/value_grad/tuple/control_dependency_1Identity0gradients/mean_squared_error/value_grad/Select_19^gradients/mean_squared_error/value_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients/mean_squared_error/value_grad/Select_1*
_output_shapes
: 
é
Dgradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/MulMulAgradients/total_regularization_loss_grad/tuple/control_dependency8layer_fc0/dense/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
ę
Fgradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1MulAgradients/total_regularization_loss_grad/tuple/control_dependency7layer_fc0/dense/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
é
Qgradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpE^gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/MulG^gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ů
Ygradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityDgradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/MulR^gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
_output_shapes
: *
T0*W
_classM
KIloc:@gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/Mul
˙
[gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityFgradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1R^gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ë
Dgradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/MulMulCgradients/total_regularization_loss_grad/tuple/control_dependency_18layer_fc1/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
ě
Fgradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1MulCgradients/total_regularization_loss_grad/tuple/control_dependency_17layer_fc1/dense/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
é
Qgradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpE^gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/MulG^gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ů
Ygradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityDgradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/MulR^gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
˙
[gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityFgradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1R^gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
ë
Dgradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/MulMulCgradients/total_regularization_loss_grad/tuple/control_dependency_28layer_fc2/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
ě
Fgradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1MulCgradients/total_regularization_loss_grad/tuple/control_dependency_27layer_fc2/dense/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
é
Qgradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpE^gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/MulG^gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ů
Ygradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityDgradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/MulR^gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
˙
[gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityFgradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1R^gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ë
Dgradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/MulMulCgradients/total_regularization_loss_grad/tuple/control_dependency_38layer_fc3/dense/kernel/Regularizer/l2_regularizer/L2Loss*
_output_shapes
: *
T0
ě
Fgradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1MulCgradients/total_regularization_loss_grad/tuple/control_dependency_37layer_fc3/dense/kernel/Regularizer/l2_regularizer/scale*
T0*
_output_shapes
: 
é
Qgradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpE^gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/MulG^gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ů
Ygradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityDgradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/MulR^gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
˙
[gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityFgradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1R^gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
ë
Dgradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/MulMulCgradients/total_regularization_loss_grad/tuple/control_dependency_48layer_fc4/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
ě
Fgradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1MulCgradients/total_regularization_loss_grad/tuple/control_dependency_47layer_fc4/dense/kernel/Regularizer/l2_regularizer/scale*
_output_shapes
: *
T0
é
Qgradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpE^gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/MulG^gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ů
Ygradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityDgradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/MulR^gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
_output_shapes
: *
T0*W
_classM
KIloc:@gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/Mul
˙
[gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityFgradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1R^gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*Y
_classO
MKloc:@gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1*
_output_shapes
: 
ë
Dgradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/MulMulCgradients/total_regularization_loss_grad/tuple/control_dependency_58layer_fc5/dense/kernel/Regularizer/l2_regularizer/L2Loss*
T0*
_output_shapes
: 
ě
Fgradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1MulCgradients/total_regularization_loss_grad/tuple/control_dependency_57layer_fc5/dense/kernel/Regularizer/l2_regularizer/scale*
_output_shapes
: *
T0
é
Qgradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_depsNoOpE^gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/MulG^gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
ů
Ygradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependencyIdentityDgradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/MulR^gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
T0*W
_classM
KIloc:@gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/Mul*
_output_shapes
: 
˙
[gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1IdentityFgradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1R^gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/tuple/group_deps*
_output_shapes
: *
T0*Y
_classO
MKloc:@gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/Mul_1
n
+gradients/mean_squared_error/div_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
p
-gradients/mean_squared_error/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
í
;gradients/mean_squared_error/div_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/div_grad/Shape-gradients/mean_squared_error/div_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ś
-gradients/mean_squared_error/div_grad/RealDivRealDiv@gradients/mean_squared_error/value_grad/tuple/control_dependencymean_squared_error/Select*
_output_shapes
: *
T0
Ú
)gradients/mean_squared_error/div_grad/SumSum-gradients/mean_squared_error/div_grad/RealDiv;gradients/mean_squared_error/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
ż
-gradients/mean_squared_error/div_grad/ReshapeReshape)gradients/mean_squared_error/div_grad/Sum+gradients/mean_squared_error/div_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
k
)gradients/mean_squared_error/div_grad/NegNegmean_squared_error/Sum_1*
T0*
_output_shapes
: 
Ą
/gradients/mean_squared_error/div_grad/RealDiv_1RealDiv)gradients/mean_squared_error/div_grad/Negmean_squared_error/Select*
_output_shapes
: *
T0
§
/gradients/mean_squared_error/div_grad/RealDiv_2RealDiv/gradients/mean_squared_error/div_grad/RealDiv_1mean_squared_error/Select*
T0*
_output_shapes
: 
Ä
)gradients/mean_squared_error/div_grad/mulMul@gradients/mean_squared_error/value_grad/tuple/control_dependency/gradients/mean_squared_error/div_grad/RealDiv_2*
T0*
_output_shapes
: 
Ú
+gradients/mean_squared_error/div_grad/Sum_1Sum)gradients/mean_squared_error/div_grad/mul=gradients/mean_squared_error/div_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
Ĺ
/gradients/mean_squared_error/div_grad/Reshape_1Reshape+gradients/mean_squared_error/div_grad/Sum_1-gradients/mean_squared_error/div_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
6gradients/mean_squared_error/div_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/div_grad/Reshape0^gradients/mean_squared_error/div_grad/Reshape_1

>gradients/mean_squared_error/div_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/div_grad/Reshape7^gradients/mean_squared_error/div_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mean_squared_error/div_grad/Reshape*
_output_shapes
: 

@gradients/mean_squared_error/div_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/div_grad/Reshape_17^gradients/mean_squared_error/div_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/div_grad/Reshape_1*
_output_shapes
: 
ő
Kgradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMullayer_fc0/dense/kernel/read[gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
_output_shapes

: *
T0
ő
Kgradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMullayer_fc1/dense/kernel/read[gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
_output_shapes

: @*
T0
ö
Kgradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMullayer_fc2/dense/kernel/read[gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0*
_output_shapes
:	@
÷
Kgradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMullayer_fc3/dense/kernel/read[gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

÷
Kgradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMullayer_fc4/dense/kernel/read[gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

÷
Kgradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mulMullayer_fc5/dense/kernel/read[gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer_grad/tuple/control_dependency_1*
T0* 
_output_shapes
:

x
5gradients/mean_squared_error/Sum_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
ŕ
/gradients/mean_squared_error/Sum_1_grad/ReshapeReshape>gradients/mean_squared_error/div_grad/tuple/control_dependency5gradients/mean_squared_error/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
: 
p
-gradients/mean_squared_error/Sum_1_grad/ConstConst*
valueB *
dtype0*
_output_shapes
: 
Ç
,gradients/mean_squared_error/Sum_1_grad/TileTile/gradients/mean_squared_error/Sum_1_grad/Reshape-gradients/mean_squared_error/Sum_1_grad/Const*

Tmultiples0*
T0*
_output_shapes
: 

+gradients/mean_squared_error/Sum_grad/ShapeShapemean_squared_error/Mul*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
out_type0
Đ
*gradients/mean_squared_error/Sum_grad/SizeSize+gradients/mean_squared_error/Sum_grad/Shape*
T0*
out_type0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
_output_shapes
: 
ä
)gradients/mean_squared_error/Sum_grad/addAddmean_squared_error/range*gradients/mean_squared_error/Sum_grad/Size*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ú
)gradients/mean_squared_error/Sum_grad/modFloorMod)gradients/mean_squared_error/Sum_grad/add*gradients/mean_squared_error/Sum_grad/Size*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ö
-gradients/mean_squared_error/Sum_grad/Shape_1Shape)gradients/mean_squared_error/Sum_grad/mod*
out_type0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
_output_shapes
:*
T0
ł
1gradients/mean_squared_error/Sum_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape
ł
1gradients/mean_squared_error/Sum_grad/range/deltaConst*
value	B :*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
dtype0*
_output_shapes
: 
ˇ
+gradients/mean_squared_error/Sum_grad/rangeRange1gradients/mean_squared_error/Sum_grad/range/start*gradients/mean_squared_error/Sum_grad/Size1gradients/mean_squared_error/Sum_grad/range/delta*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape
˛
0gradients/mean_squared_error/Sum_grad/Fill/valueConst*
_output_shapes
: *
value	B :*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
dtype0

*gradients/mean_squared_error/Sum_grad/FillFill-gradients/mean_squared_error/Sum_grad/Shape_10gradients/mean_squared_error/Sum_grad/Fill/value*
T0*

index_type0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
ě
3gradients/mean_squared_error/Sum_grad/DynamicStitchDynamicStitch+gradients/mean_squared_error/Sum_grad/range)gradients/mean_squared_error/Sum_grad/mod+gradients/mean_squared_error/Sum_grad/Shape*gradients/mean_squared_error/Sum_grad/Fill*
N*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape
ą
/gradients/mean_squared_error/Sum_grad/Maximum/yConst*
value	B :*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*
dtype0*
_output_shapes
: 

-gradients/mean_squared_error/Sum_grad/MaximumMaximum3gradients/mean_squared_error/Sum_grad/DynamicStitch/gradients/mean_squared_error/Sum_grad/Maximum/y*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

.gradients/mean_squared_error/Sum_grad/floordivFloorDiv+gradients/mean_squared_error/Sum_grad/Shape-gradients/mean_squared_error/Sum_grad/Maximum*
T0*>
_class4
20loc:@gradients/mean_squared_error/Sum_grad/Shape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ě
-gradients/mean_squared_error/Sum_grad/ReshapeReshape,gradients/mean_squared_error/Sum_1_grad/Tile3gradients/mean_squared_error/Sum_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
Ć
*gradients/mean_squared_error/Sum_grad/TileTile-gradients/mean_squared_error/Sum_grad/Reshape.gradients/mean_squared_error/Sum_grad/floordiv*

Tmultiples0*
T0*
_output_shapes
:

+gradients/mean_squared_error/Mul_grad/ShapeShape$mean_squared_error/SquaredDifference*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
out_type0
p
-gradients/mean_squared_error/Mul_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
í
;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgsBroadcastGradientArgs+gradients/mean_squared_error/Mul_grad/Shape-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙

)gradients/mean_squared_error/Mul_grad/MulMul*gradients/mean_squared_error/Sum_grad/Tilemean_squared_error/ToFloat/x*
T0*
_output_shapes
:
Ř
)gradients/mean_squared_error/Mul_grad/SumSum)gradients/mean_squared_error/Mul_grad/Mul;gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Á
-gradients/mean_squared_error/Mul_grad/ReshapeReshape)gradients/mean_squared_error/Mul_grad/Sum+gradients/mean_squared_error/Mul_grad/Shape*
_output_shapes
:*
T0*
Tshape0
§
+gradients/mean_squared_error/Mul_grad/Mul_1Mul$mean_squared_error/SquaredDifference*gradients/mean_squared_error/Sum_grad/Tile*
_output_shapes
:*
T0
Ţ
+gradients/mean_squared_error/Mul_grad/Sum_1Sum+gradients/mean_squared_error/Mul_grad/Mul_1=gradients/mean_squared_error/Mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ĺ
/gradients/mean_squared_error/Mul_grad/Reshape_1Reshape+gradients/mean_squared_error/Mul_grad/Sum_1-gradients/mean_squared_error/Mul_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
 
6gradients/mean_squared_error/Mul_grad/tuple/group_depsNoOp.^gradients/mean_squared_error/Mul_grad/Reshape0^gradients/mean_squared_error/Mul_grad/Reshape_1

>gradients/mean_squared_error/Mul_grad/tuple/control_dependencyIdentity-gradients/mean_squared_error/Mul_grad/Reshape7^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*@
_class6
42loc:@gradients/mean_squared_error/Mul_grad/Reshape*
_output_shapes
:

@gradients/mean_squared_error/Mul_grad/tuple/control_dependency_1Identity/gradients/mean_squared_error/Mul_grad/Reshape_17^gradients/mean_squared_error/Mul_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/mean_squared_error/Mul_grad/Reshape_1*
_output_shapes
: 
~
9gradients/mean_squared_error/SquaredDifference_grad/ShapeShapeLog_1*
_output_shapes
:*
T0*
out_type0

;gradients/mean_squared_error/SquaredDifference_grad/Shape_1ShapeLog*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
out_type0

Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgsBroadcastGradientArgs9gradients/mean_squared_error/SquaredDifference_grad/Shape;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
Ŕ
:gradients/mean_squared_error/SquaredDifference_grad/scalarConst?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
valueB
 *   @*
dtype0*
_output_shapes
: 
Ý
7gradients/mean_squared_error/SquaredDifference_grad/mulMul:gradients/mean_squared_error/SquaredDifference_grad/scalar>gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
_output_shapes
:*
T0
Ž
7gradients/mean_squared_error/SquaredDifference_grad/subSubLog_1Log?^gradients/mean_squared_error/Mul_grad/tuple/control_dependency*
T0*
_output_shapes
:
Ő
9gradients/mean_squared_error/SquaredDifference_grad/mul_1Mul7gradients/mean_squared_error/SquaredDifference_grad/mul7gradients/mean_squared_error/SquaredDifference_grad/sub*
_output_shapes
:*
T0

7gradients/mean_squared_error/SquaredDifference_grad/SumSum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Igradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
ö
;gradients/mean_squared_error/SquaredDifference_grad/ReshapeReshape7gradients/mean_squared_error/SquaredDifference_grad/Sum9gradients/mean_squared_error/SquaredDifference_grad/Shape*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙

9gradients/mean_squared_error/SquaredDifference_grad/Sum_1Sum9gradients/mean_squared_error/SquaredDifference_grad/mul_1Kgradients/mean_squared_error/SquaredDifference_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
ń
=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1Reshape9gradients/mean_squared_error/SquaredDifference_grad/Sum_1;gradients/mean_squared_error/SquaredDifference_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
 
7gradients/mean_squared_error/SquaredDifference_grad/NegNeg=gradients/mean_squared_error/SquaredDifference_grad/Reshape_1*
T0*
_output_shapes
:
Ä
Dgradients/mean_squared_error/SquaredDifference_grad/tuple/group_depsNoOp8^gradients/mean_squared_error/SquaredDifference_grad/Neg<^gradients/mean_squared_error/SquaredDifference_grad/Reshape
Ú
Lgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencyIdentity;gradients/mean_squared_error/SquaredDifference_grad/ReshapeE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*
T0*N
_classD
B@loc:@gradients/mean_squared_error/SquaredDifference_grad/Reshape*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
Ngradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency_1Identity7gradients/mean_squared_error/SquaredDifference_grad/NegE^gradients/mean_squared_error/SquaredDifference_grad/tuple/group_deps*J
_class@
><loc:@gradients/mean_squared_error/SquaredDifference_grad/Neg*
_output_shapes
:*
T0
ą
gradients/Log_1_grad/Reciprocal
Reciprocaladd_1M^gradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependency*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ź
gradients/Log_1_grad/mulMulLgradients/mean_squared_error/SquaredDifference_grad/tuple/control_dependencygradients/Log_1_grad/Reciprocal*
T0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
]
gradients/add_1_grad/ShapeConst*
dtype0*
_output_shapes
: *
valueB 
l
gradients/add_1_grad/Shape_1Shapemodel_prediction*
T0*
out_type0*
_output_shapes
:
ş
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ľ
gradients/add_1_grad/SumSumgradients/Log_1_grad/mul*gradients/add_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
Š
gradients/add_1_grad/Sum_1Sumgradients/Log_1_grad/mul,gradients/add_1_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 

gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:˙˙˙˙˙˙˙˙˙
m
%gradients/add_1_grad/tuple/group_depsNoOp^gradients/add_1_grad/Reshape^gradients/add_1_grad/Reshape_1
Ń
-gradients/add_1_grad/tuple/control_dependencyIdentitygradients/add_1_grad/Reshape&^gradients/add_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/add_1_grad/Reshape*
_output_shapes
: 
ä
/gradients/add_1_grad/tuple/control_dependency_1Identitygradients/add_1_grad/Reshape_1&^gradients/add_1_grad/tuple/group_deps*1
_class'
%#loc:@gradients/add_1_grad/Reshape_1*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
o
%gradients/model_prediction_grad/ShapeShape
dense/Relu*
T0*
out_type0*
_output_shapes
:
Ę
'gradients/model_prediction_grad/ReshapeReshape/gradients/add_1_grad/tuple/control_dependency_1%gradients/model_prediction_grad/Shape*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0

"gradients/dense/Relu_grad/ReluGradReluGrad'gradients/model_prediction_grad/Reshape
dense/Relu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ź
"gradients/dense/MatMul_grad/MatMulMatMul"gradients/dense/Relu_grad/ReluGraddense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
ś
$gradients/dense/MatMul_grad/MatMul_1MatMuldropout/cond/Merge"gradients/dense/Relu_grad/ReluGrad*
T0*
transpose_a(*
_output_shapes
:	*
transpose_b( 

,gradients/dense/MatMul_grad/tuple/group_depsNoOp#^gradients/dense/MatMul_grad/MatMul%^gradients/dense/MatMul_grad/MatMul_1
ý
4gradients/dense/MatMul_grad/tuple/control_dependencyIdentity"gradients/dense/MatMul_grad/MatMul-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ú
6gradients/dense/MatMul_grad/tuple/control_dependency_1Identity$gradients/dense/MatMul_grad/MatMul_1-^gradients/dense/MatMul_grad/tuple/group_deps*
T0*7
_class-
+)loc:@gradients/dense/MatMul_grad/MatMul_1*
_output_shapes
:	
˙
+gradients/dropout/cond/Merge_grad/cond_gradSwitch4gradients/dense/MatMul_grad/tuple/control_dependencydropout/cond/pred_id*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
h
2gradients/dropout/cond/Merge_grad/tuple/group_depsNoOp,^gradients/dropout/cond/Merge_grad/cond_grad

:gradients/dropout/cond/Merge_grad/tuple/control_dependencyIdentity+gradients/dropout/cond/Merge_grad/cond_grad3^gradients/dropout/cond/Merge_grad/tuple/group_deps*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0

<gradients/dropout/cond/Merge_grad/tuple/control_dependency_1Identity-gradients/dropout/cond/Merge_grad/cond_grad:13^gradients/dropout/cond/Merge_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

-gradients/dropout/cond/dropout/mul_grad/ShapeShapedropout/cond/dropout/div*
T0*
out_type0*
_output_shapes
:

/gradients/dropout/cond/dropout/mul_grad/Shape_1Shapedropout/cond/dropout/Floor*
T0*
out_type0*
_output_shapes
:
ó
=gradients/dropout/cond/dropout/mul_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/dropout/cond/dropout/mul_grad/Shape/gradients/dropout/cond/dropout/mul_grad/Shape_1*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
ż
+gradients/dropout/cond/dropout/mul_grad/MulMul<gradients/dropout/cond/Merge_grad/tuple/control_dependency_1dropout/cond/dropout/Floor*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
Ţ
+gradients/dropout/cond/dropout/mul_grad/SumSum+gradients/dropout/cond/dropout/mul_grad/Mul=gradients/dropout/cond/dropout/mul_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
×
/gradients/dropout/cond/dropout/mul_grad/ReshapeReshape+gradients/dropout/cond/dropout/mul_grad/Sum-gradients/dropout/cond/dropout/mul_grad/Shape*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ż
-gradients/dropout/cond/dropout/mul_grad/Mul_1Muldropout/cond/dropout/div<gradients/dropout/cond/Merge_grad/tuple/control_dependency_1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ä
-gradients/dropout/cond/dropout/mul_grad/Sum_1Sum-gradients/dropout/cond/dropout/mul_grad/Mul_1?gradients/dropout/cond/dropout/mul_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
Ý
1gradients/dropout/cond/dropout/mul_grad/Reshape_1Reshape-gradients/dropout/cond/dropout/mul_grad/Sum_1/gradients/dropout/cond/dropout/mul_grad/Shape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*
Tshape0
Ś
8gradients/dropout/cond/dropout/mul_grad/tuple/group_depsNoOp0^gradients/dropout/cond/dropout/mul_grad/Reshape2^gradients/dropout/cond/dropout/mul_grad/Reshape_1
Ż
@gradients/dropout/cond/dropout/mul_grad/tuple/control_dependencyIdentity/gradients/dropout/cond/dropout/mul_grad/Reshape9^gradients/dropout/cond/dropout/mul_grad/tuple/group_deps*B
_class8
64loc:@gradients/dropout/cond/dropout/mul_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ľ
Bgradients/dropout/cond/dropout/mul_grad/tuple/control_dependency_1Identity1gradients/dropout/cond/dropout/mul_grad/Reshape_19^gradients/dropout/cond/dropout/mul_grad/tuple/group_deps*
T0*D
_class:
86loc:@gradients/dropout/cond/dropout/mul_grad/Reshape_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

gradients/SwitchSwitchlayer_fc5/dense/Reludropout/cond/pred_id*
T0*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
e
gradients/IdentityIdentitygradients/Switch:1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
gradients/Shape_1Shapegradients/Switch:1*
T0*
out_type0*
_output_shapes
:
o
gradients/zeros/ConstConst^gradients/Identity*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zerosFillgradients/Shape_1gradients/zeros/Const*
T0*

index_type0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
É
5gradients/dropout/cond/Identity/Switch_grad/cond_gradMerge:gradients/dropout/cond/Merge_grad/tuple/control_dependencygradients/zeros*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 

-gradients/dropout/cond/dropout/div_grad/ShapeShape#dropout/cond/dropout/Shape/Switch:1*
T0*
out_type0*
_output_shapes
:
r
/gradients/dropout/cond/dropout/div_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
ó
=gradients/dropout/cond/dropout/div_grad/BroadcastGradientArgsBroadcastGradientArgs-gradients/dropout/cond/dropout/div_grad/Shape/gradients/dropout/cond/dropout/div_grad/Shape_1*
T0*2
_output_shapes 
:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙
Ď
/gradients/dropout/cond/dropout/div_grad/RealDivRealDiv@gradients/dropout/cond/dropout/mul_grad/tuple/control_dependencydropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
+gradients/dropout/cond/dropout/div_grad/SumSum/gradients/dropout/cond/dropout/div_grad/RealDiv=gradients/dropout/cond/dropout/div_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
×
/gradients/dropout/cond/dropout/div_grad/ReshapeReshape+gradients/dropout/cond/dropout/div_grad/Sum-gradients/dropout/cond/dropout/div_grad/Shape*
T0*
Tshape0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

+gradients/dropout/cond/dropout/div_grad/NegNeg#dropout/cond/dropout/Shape/Switch:1*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ź
1gradients/dropout/cond/dropout/div_grad/RealDiv_1RealDiv+gradients/dropout/cond/dropout/div_grad/Negdropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Â
1gradients/dropout/cond/dropout/div_grad/RealDiv_2RealDiv1gradients/dropout/cond/dropout/div_grad/RealDiv_1dropout/cond/dropout/keep_prob*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ú
+gradients/dropout/cond/dropout/div_grad/mulMul@gradients/dropout/cond/dropout/mul_grad/tuple/control_dependency1gradients/dropout/cond/dropout/div_grad/RealDiv_2*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
â
-gradients/dropout/cond/dropout/div_grad/Sum_1Sum+gradients/dropout/cond/dropout/div_grad/mul?gradients/dropout/cond/dropout/div_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
Ë
1gradients/dropout/cond/dropout/div_grad/Reshape_1Reshape-gradients/dropout/cond/dropout/div_grad/Sum_1/gradients/dropout/cond/dropout/div_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
Ś
8gradients/dropout/cond/dropout/div_grad/tuple/group_depsNoOp0^gradients/dropout/cond/dropout/div_grad/Reshape2^gradients/dropout/cond/dropout/div_grad/Reshape_1
Ż
@gradients/dropout/cond/dropout/div_grad/tuple/control_dependencyIdentity/gradients/dropout/cond/dropout/div_grad/Reshape9^gradients/dropout/cond/dropout/div_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients/dropout/cond/dropout/div_grad/Reshape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
Bgradients/dropout/cond/dropout/div_grad/tuple/control_dependency_1Identity1gradients/dropout/cond/dropout/div_grad/Reshape_19^gradients/dropout/cond/dropout/div_grad/tuple/group_deps*D
_class:
86loc:@gradients/dropout/cond/dropout/div_grad/Reshape_1*
_output_shapes
: *
T0

gradients/Switch_1Switchlayer_fc5/dense/Reludropout/cond/pred_id*<
_output_shapes*
(:˙˙˙˙˙˙˙˙˙:˙˙˙˙˙˙˙˙˙*
T0
g
gradients/Identity_1Identitygradients/Switch_1*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
c
gradients/Shape_2Shapegradients/Switch_1*
_output_shapes
:*
T0*
out_type0
s
gradients/zeros_1/ConstConst^gradients/Identity_1*
valueB
 *    *
dtype0*
_output_shapes
: 

gradients/zeros_1Fillgradients/Shape_2gradients/zeros_1/Const*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*

index_type0
Ö
:gradients/dropout/cond/dropout/Shape/Switch_grad/cond_gradMergegradients/zeros_1@gradients/dropout/cond/dropout/div_grad/tuple/control_dependency*
T0*
N**
_output_shapes
:˙˙˙˙˙˙˙˙˙: 

gradients/AddNAddN5gradients/dropout/cond/Identity/Switch_grad/cond_grad:gradients/dropout/cond/dropout/Shape/Switch_grad/cond_grad*
N*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*H
_class>
<:loc:@gradients/dropout/cond/Identity/Switch_grad/cond_grad

,gradients/layer_fc5/dense/Relu_grad/ReluGradReluGradgradients/AddNlayer_fc5/dense/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ź
2gradients/layer_fc5/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/layer_fc5/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/layer_fc5/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/layer_fc5/dense/BiasAdd_grad/BiasAddGrad-^gradients/layer_fc5/dense/Relu_grad/ReluGrad
§
?gradients/layer_fc5/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/layer_fc5/dense/Relu_grad/ReluGrad8^gradients/layer_fc5/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/layer_fc5/dense/Relu_grad/ReluGrad
¨
Agradients/layer_fc5/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/layer_fc5/dense/BiasAdd_grad/BiasAddGrad8^gradients/layer_fc5/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_fc5/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
í
,gradients/layer_fc5/dense/MatMul_grad/MatMulMatMul?gradients/layer_fc5/dense/BiasAdd_grad/tuple/control_dependencylayer_fc5/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
ŕ
.gradients/layer_fc5/dense/MatMul_grad/MatMul_1MatMullayer_fc4/dense/Relu?gradients/layer_fc5/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(* 
_output_shapes
:
*
transpose_b( *
T0

6gradients/layer_fc5/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/layer_fc5/dense/MatMul_grad/MatMul/^gradients/layer_fc5/dense/MatMul_grad/MatMul_1
Ľ
>gradients/layer_fc5/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/layer_fc5/dense/MatMul_grad/MatMul7^gradients/layer_fc5/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc5/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/layer_fc5/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/layer_fc5/dense/MatMul_grad/MatMul_17^gradients/layer_fc5/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/layer_fc5/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients/layer_fc4/dense/Relu_grad/ReluGradReluGrad>gradients/layer_fc5/dense/MatMul_grad/tuple/control_dependencylayer_fc4/dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
gradients/AddN_1AddNKgradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul@gradients/layer_fc5/dense/MatMul_grad/tuple/control_dependency_1* 
_output_shapes
:
*
T0*^
_classT
RPloc:@gradients/layer_fc5/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
N
Ź
2gradients/layer_fc4/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/layer_fc4/dense/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
Ł
7gradients/layer_fc4/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/layer_fc4/dense/BiasAdd_grad/BiasAddGrad-^gradients/layer_fc4/dense/Relu_grad/ReluGrad
§
?gradients/layer_fc4/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/layer_fc4/dense/Relu_grad/ReluGrad8^gradients/layer_fc4/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc4/dense/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Agradients/layer_fc4/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/layer_fc4/dense/BiasAdd_grad/BiasAddGrad8^gradients/layer_fc4/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_fc4/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
í
,gradients/layer_fc4/dense/MatMul_grad/MatMulMatMul?gradients/layer_fc4/dense/BiasAdd_grad/tuple/control_dependencylayer_fc4/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
ŕ
.gradients/layer_fc4/dense/MatMul_grad/MatMul_1MatMullayer_fc3/dense/Relu?gradients/layer_fc4/dense/BiasAdd_grad/tuple/control_dependency*
T0*
transpose_a(* 
_output_shapes
:
*
transpose_b( 

6gradients/layer_fc4/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/layer_fc4/dense/MatMul_grad/MatMul/^gradients/layer_fc4/dense/MatMul_grad/MatMul_1
Ľ
>gradients/layer_fc4/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/layer_fc4/dense/MatMul_grad/MatMul7^gradients/layer_fc4/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc4/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/layer_fc4/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/layer_fc4/dense/MatMul_grad/MatMul_17^gradients/layer_fc4/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/layer_fc4/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients/layer_fc3/dense/Relu_grad/ReluGradReluGrad>gradients/layer_fc4/dense/MatMul_grad/tuple/control_dependencylayer_fc3/dense/Relu*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0
ť
gradients/AddN_2AddNKgradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul@gradients/layer_fc4/dense/MatMul_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/layer_fc4/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
N* 
_output_shapes
:

Ź
2gradients/layer_fc3/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/layer_fc3/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes	
:
Ł
7gradients/layer_fc3/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/layer_fc3/dense/BiasAdd_grad/BiasAddGrad-^gradients/layer_fc3/dense/Relu_grad/ReluGrad
§
?gradients/layer_fc3/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/layer_fc3/dense/Relu_grad/ReluGrad8^gradients/layer_fc3/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc3/dense/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Agradients/layer_fc3/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/layer_fc3/dense/BiasAdd_grad/BiasAddGrad8^gradients/layer_fc3/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/layer_fc3/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:*
T0
í
,gradients/layer_fc3/dense/MatMul_grad/MatMulMatMul?gradients/layer_fc3/dense/BiasAdd_grad/tuple/control_dependencylayer_fc3/dense/kernel/read*
transpose_a( *(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
ŕ
.gradients/layer_fc3/dense/MatMul_grad/MatMul_1MatMullayer_fc2/dense/Relu?gradients/layer_fc3/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:


6gradients/layer_fc3/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/layer_fc3/dense/MatMul_grad/MatMul/^gradients/layer_fc3/dense/MatMul_grad/MatMul_1
Ľ
>gradients/layer_fc3/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/layer_fc3/dense/MatMul_grad/MatMul7^gradients/layer_fc3/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc3/dense/MatMul_grad/MatMul*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
@gradients/layer_fc3/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/layer_fc3/dense/MatMul_grad/MatMul_17^gradients/layer_fc3/dense/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients/layer_fc3/dense/MatMul_grad/MatMul_1* 
_output_shapes
:

Á
,gradients/layer_fc2/dense/Relu_grad/ReluGradReluGrad>gradients/layer_fc3/dense/MatMul_grad/tuple/control_dependencylayer_fc2/dense/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
ť
gradients/AddN_3AddNKgradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul@gradients/layer_fc3/dense/MatMul_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/layer_fc3/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
N* 
_output_shapes
:

Ź
2gradients/layer_fc2/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/layer_fc2/dense/Relu_grad/ReluGrad*
_output_shapes	
:*
T0*
data_formatNHWC
Ł
7gradients/layer_fc2/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/layer_fc2/dense/BiasAdd_grad/BiasAddGrad-^gradients/layer_fc2/dense/Relu_grad/ReluGrad
§
?gradients/layer_fc2/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/layer_fc2/dense/Relu_grad/ReluGrad8^gradients/layer_fc2/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc2/dense/Relu_grad/ReluGrad*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
¨
Agradients/layer_fc2/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/layer_fc2/dense/BiasAdd_grad/BiasAddGrad8^gradients/layer_fc2/dense/BiasAdd_grad/tuple/group_deps*
T0*E
_class;
97loc:@gradients/layer_fc2/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:
ě
,gradients/layer_fc2/dense/MatMul_grad/MatMulMatMul?gradients/layer_fc2/dense/BiasAdd_grad/tuple/control_dependencylayer_fc2/dense/kernel/read*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
transpose_b(
ß
.gradients/layer_fc2/dense/MatMul_grad/MatMul_1MatMullayer_fc1/dense/Relu?gradients/layer_fc2/dense/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
transpose_a(*
_output_shapes
:	@

6gradients/layer_fc2/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/layer_fc2/dense/MatMul_grad/MatMul/^gradients/layer_fc2/dense/MatMul_grad/MatMul_1
¤
>gradients/layer_fc2/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/layer_fc2/dense/MatMul_grad/MatMul7^gradients/layer_fc2/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc2/dense/MatMul_grad/MatMul*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
˘
@gradients/layer_fc2/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/layer_fc2/dense/MatMul_grad/MatMul_17^gradients/layer_fc2/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	@*
T0*A
_class7
53loc:@gradients/layer_fc2/dense/MatMul_grad/MatMul_1
Ŕ
,gradients/layer_fc1/dense/Relu_grad/ReluGradReluGrad>gradients/layer_fc2/dense/MatMul_grad/tuple/control_dependencylayer_fc1/dense/Relu*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ş
gradients/AddN_4AddNKgradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul@gradients/layer_fc2/dense/MatMul_grad/tuple/control_dependency_1*^
_classT
RPloc:@gradients/layer_fc2/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
N*
_output_shapes
:	@*
T0
Ť
2gradients/layer_fc1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/layer_fc1/dense/Relu_grad/ReluGrad*
_output_shapes
:@*
T0*
data_formatNHWC
Ł
7gradients/layer_fc1/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/layer_fc1/dense/BiasAdd_grad/BiasAddGrad-^gradients/layer_fc1/dense/Relu_grad/ReluGrad
Ś
?gradients/layer_fc1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/layer_fc1/dense/Relu_grad/ReluGrad8^gradients/layer_fc1/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc1/dense/Relu_grad/ReluGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙@
§
Agradients/layer_fc1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/layer_fc1/dense/BiasAdd_grad/BiasAddGrad8^gradients/layer_fc1/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*E
_class;
97loc:@gradients/layer_fc1/dense/BiasAdd_grad/BiasAddGrad
ě
,gradients/layer_fc1/dense/MatMul_grad/MatMulMatMul?gradients/layer_fc1/dense/BiasAdd_grad/tuple/control_dependencylayer_fc1/dense/kernel/read*
transpose_b(*
T0*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
Ţ
.gradients/layer_fc1/dense/MatMul_grad/MatMul_1MatMullayer_fc0/dense/Relu?gradients/layer_fc1/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

: @*
transpose_b( *
T0

6gradients/layer_fc1/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/layer_fc1/dense/MatMul_grad/MatMul/^gradients/layer_fc1/dense/MatMul_grad/MatMul_1
¤
>gradients/layer_fc1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/layer_fc1/dense/MatMul_grad/MatMul7^gradients/layer_fc1/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0*?
_class5
31loc:@gradients/layer_fc1/dense/MatMul_grad/MatMul
Ą
@gradients/layer_fc1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/layer_fc1/dense/MatMul_grad/MatMul_17^gradients/layer_fc1/dense/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients/layer_fc1/dense/MatMul_grad/MatMul_1*
_output_shapes

: @*
T0
Ŕ
,gradients/layer_fc0/dense/Relu_grad/ReluGradReluGrad>gradients/layer_fc1/dense/MatMul_grad/tuple/control_dependencylayer_fc0/dense/Relu*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
T0
š
gradients/AddN_5AddNKgradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul@gradients/layer_fc1/dense/MatMul_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/layer_fc1/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
N*
_output_shapes

: @
Ť
2gradients/layer_fc0/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients/layer_fc0/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
: *
T0
Ł
7gradients/layer_fc0/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients/layer_fc0/dense/BiasAdd_grad/BiasAddGrad-^gradients/layer_fc0/dense/Relu_grad/ReluGrad
Ś
?gradients/layer_fc0/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients/layer_fc0/dense/Relu_grad/ReluGrad8^gradients/layer_fc0/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients/layer_fc0/dense/Relu_grad/ReluGrad*'
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
§
Agradients/layer_fc0/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients/layer_fc0/dense/BiasAdd_grad/BiasAddGrad8^gradients/layer_fc0/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients/layer_fc0/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
: *
T0
ě
,gradients/layer_fc0/dense/MatMul_grad/MatMulMatMul?gradients/layer_fc0/dense/BiasAdd_grad/tuple/control_dependencylayer_fc0/dense/kernel/read*
transpose_a( *'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
transpose_b(*
T0
Ő
.gradients/layer_fc0/dense/MatMul_grad/MatMul_1MatMulmodel_input?gradients/layer_fc0/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
_output_shapes

: *
transpose_b( *
T0

6gradients/layer_fc0/dense/MatMul_grad/tuple/group_depsNoOp-^gradients/layer_fc0/dense/MatMul_grad/MatMul/^gradients/layer_fc0/dense/MatMul_grad/MatMul_1
¤
>gradients/layer_fc0/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients/layer_fc0/dense/MatMul_grad/MatMul7^gradients/layer_fc0/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:˙˙˙˙˙˙˙˙˙*
T0*?
_class5
31loc:@gradients/layer_fc0/dense/MatMul_grad/MatMul
Ą
@gradients/layer_fc0/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients/layer_fc0/dense/MatMul_grad/MatMul_17^gradients/layer_fc0/dense/MatMul_grad/tuple/group_deps*
_output_shapes

: *
T0*A
_class7
53loc:@gradients/layer_fc0/dense/MatMul_grad/MatMul_1
š
gradients/AddN_6AddNKgradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul@gradients/layer_fc0/dense/MatMul_grad/tuple/control_dependency_1*
T0*^
_classT
RPloc:@gradients/layer_fc0/dense/kernel/Regularizer/l2_regularizer/L2Loss_grad/mul*
N*
_output_shapes

: 

beta1_power/initial_valueConst*
valueB
 *fff?*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

beta1_power
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape: *
dtype0*
_output_shapes
: 
Ż
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: 
k
beta1_power/readIdentitybeta1_power*
_output_shapes
: *
T0*
_class
loc:@dense/kernel

beta2_power/initial_valueConst*
valueB
 *wž?*
_class
loc:@dense/kernel*
dtype0*
_output_shapes
: 

beta2_power
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape: *
dtype0*
_output_shapes
: 
Ż
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
k
beta2_power/readIdentitybeta2_power*
_output_shapes
: *
T0*
_class
loc:@dense/kernel
­
-layer_fc0/dense/kernel/Adam/Initializer/zerosConst*
valueB *    *)
_class
loc:@layer_fc0/dense/kernel*
dtype0*
_output_shapes

: 
ş
layer_fc0/dense/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@layer_fc0/dense/kernel*
	container *
shape
: *
dtype0*
_output_shapes

: 
ő
"layer_fc0/dense/kernel/Adam/AssignAssignlayer_fc0/dense/kernel/Adam-layer_fc0/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 

 layer_fc0/dense/kernel/Adam/readIdentitylayer_fc0/dense/kernel/Adam*
_output_shapes

: *
T0*)
_class
loc:@layer_fc0/dense/kernel
Ż
/layer_fc0/dense/kernel/Adam_1/Initializer/zerosConst*
_output_shapes

: *
valueB *    *)
_class
loc:@layer_fc0/dense/kernel*
dtype0
ź
layer_fc0/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

: *
shared_name *)
_class
loc:@layer_fc0/dense/kernel*
	container *
shape
: 
ű
$layer_fc0/dense/kernel/Adam_1/AssignAssignlayer_fc0/dense/kernel/Adam_1/layer_fc0/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 
Ą
"layer_fc0/dense/kernel/Adam_1/readIdentitylayer_fc0/dense/kernel/Adam_1*
T0*)
_class
loc:@layer_fc0/dense/kernel*
_output_shapes

: 
Ą
+layer_fc0/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
: *
valueB *    *'
_class
loc:@layer_fc0/dense/bias
Ž
layer_fc0/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
: *
shared_name *'
_class
loc:@layer_fc0/dense/bias*
	container *
shape: 
é
 layer_fc0/dense/bias/Adam/AssignAssignlayer_fc0/dense/bias/Adam+layer_fc0/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(*
_output_shapes
: 

layer_fc0/dense/bias/Adam/readIdentitylayer_fc0/dense/bias/Adam*
T0*'
_class
loc:@layer_fc0/dense/bias*
_output_shapes
: 
Ł
-layer_fc0/dense/bias/Adam_1/Initializer/zerosConst*
valueB *    *'
_class
loc:@layer_fc0/dense/bias*
dtype0*
_output_shapes
: 
°
layer_fc0/dense/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@layer_fc0/dense/bias*
	container *
shape: *
dtype0*
_output_shapes
: 
ď
"layer_fc0/dense/bias/Adam_1/AssignAssignlayer_fc0/dense/bias/Adam_1-layer_fc0/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(*
_output_shapes
: 

 layer_fc0/dense/bias/Adam_1/readIdentitylayer_fc0/dense/bias/Adam_1*
T0*'
_class
loc:@layer_fc0/dense/bias*
_output_shapes
: 
š
=layer_fc1/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"    @   *)
_class
loc:@layer_fc1/dense/kernel*
dtype0*
_output_shapes
:
Ł
3layer_fc1/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc1/dense/kernel*
dtype0*
_output_shapes
: 

-layer_fc1/dense/kernel/Adam/Initializer/zerosFill=layer_fc1/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3layer_fc1/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc1/dense/kernel*
_output_shapes

: @
ş
layer_fc1/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes

: @*
shared_name *)
_class
loc:@layer_fc1/dense/kernel*
	container *
shape
: @
ő
"layer_fc1/dense/kernel/Adam/AssignAssignlayer_fc1/dense/kernel/Adam-layer_fc1/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @

 layer_fc1/dense/kernel/Adam/readIdentitylayer_fc1/dense/kernel/Adam*
T0*)
_class
loc:@layer_fc1/dense/kernel*
_output_shapes

: @
ť
?layer_fc1/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"    @   *)
_class
loc:@layer_fc1/dense/kernel*
dtype0
Ľ
5layer_fc1/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
valueB
 *    *)
_class
loc:@layer_fc1/dense/kernel*
dtype0

/layer_fc1/dense/kernel/Adam_1/Initializer/zerosFill?layer_fc1/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5layer_fc1/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc1/dense/kernel*
_output_shapes

: @
ź
layer_fc1/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

: @*
shared_name *)
_class
loc:@layer_fc1/dense/kernel*
	container *
shape
: @
ű
$layer_fc1/dense/kernel/Adam_1/AssignAssignlayer_fc1/dense/kernel/Adam_1/layer_fc1/dense/kernel/Adam_1/Initializer/zeros*
_output_shapes

: @*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(
Ą
"layer_fc1/dense/kernel/Adam_1/readIdentitylayer_fc1/dense/kernel/Adam_1*
_output_shapes

: @*
T0*)
_class
loc:@layer_fc1/dense/kernel
Ą
+layer_fc1/dense/bias/Adam/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *'
_class
loc:@layer_fc1/dense/bias*
dtype0
Ž
layer_fc1/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@layer_fc1/dense/bias*
	container *
shape:@
é
 layer_fc1/dense/bias/Adam/AssignAssignlayer_fc1/dense/bias/Adam+layer_fc1/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@

layer_fc1/dense/bias/Adam/readIdentitylayer_fc1/dense/bias/Adam*
T0*'
_class
loc:@layer_fc1/dense/bias*
_output_shapes
:@
Ł
-layer_fc1/dense/bias/Adam_1/Initializer/zerosConst*
valueB@*    *'
_class
loc:@layer_fc1/dense/bias*
dtype0*
_output_shapes
:@
°
layer_fc1/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@layer_fc1/dense/bias*
	container *
shape:@
ď
"layer_fc1/dense/bias/Adam_1/AssignAssignlayer_fc1/dense/bias/Adam_1-layer_fc1/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@

 layer_fc1/dense/bias/Adam_1/readIdentitylayer_fc1/dense/bias/Adam_1*'
_class
loc:@layer_fc1/dense/bias*
_output_shapes
:@*
T0
š
=layer_fc2/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"@      *)
_class
loc:@layer_fc2/dense/kernel*
dtype0*
_output_shapes
:
Ł
3layer_fc2/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc2/dense/kernel*
dtype0*
_output_shapes
: 

-layer_fc2/dense/kernel/Adam/Initializer/zerosFill=layer_fc2/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3layer_fc2/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
:	@
ź
layer_fc2/dense/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@layer_fc2/dense/kernel*
	container *
shape:	@*
dtype0*
_output_shapes
:	@
ö
"layer_fc2/dense/kernel/Adam/AssignAssignlayer_fc2/dense/kernel/Adam-layer_fc2/dense/kernel/Adam/Initializer/zeros*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(

 layer_fc2/dense/kernel/Adam/readIdentitylayer_fc2/dense/kernel/Adam*
T0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
:	@
ť
?layer_fc2/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"@      *)
_class
loc:@layer_fc2/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc2/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc2/dense/kernel*
dtype0*
_output_shapes
: 

/layer_fc2/dense/kernel/Adam_1/Initializer/zerosFill?layer_fc2/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5layer_fc2/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
:	@
ž
layer_fc2/dense/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@layer_fc2/dense/kernel*
	container *
shape:	@*
dtype0*
_output_shapes
:	@
ü
$layer_fc2/dense/kernel/Adam_1/AssignAssignlayer_fc2/dense/kernel/Adam_1/layer_fc2/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@
˘
"layer_fc2/dense/kernel/Adam_1/readIdentitylayer_fc2/dense/kernel/Adam_1*
T0*)
_class
loc:@layer_fc2/dense/kernel*
_output_shapes
:	@
Ł
+layer_fc2/dense/bias/Adam/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *'
_class
loc:@layer_fc2/dense/bias*
dtype0
°
layer_fc2/dense/bias/Adam
VariableV2*
shared_name *'
_class
loc:@layer_fc2/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
ę
 layer_fc2/dense/bias/Adam/AssignAssignlayer_fc2/dense/bias/Adam+layer_fc2/dense/bias/Adam/Initializer/zeros*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0

layer_fc2/dense/bias/Adam/readIdentitylayer_fc2/dense/bias/Adam*
_output_shapes	
:*
T0*'
_class
loc:@layer_fc2/dense/bias
Ľ
-layer_fc2/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc2/dense/bias*
dtype0*
_output_shapes	
:
˛
layer_fc2/dense/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@layer_fc2/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:
đ
"layer_fc2/dense/bias/Adam_1/AssignAssignlayer_fc2/dense/bias/Adam_1-layer_fc2/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:

 layer_fc2/dense/bias/Adam_1/readIdentitylayer_fc2/dense/bias/Adam_1*'
_class
loc:@layer_fc2/dense/bias*
_output_shapes	
:*
T0
š
=layer_fc3/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@layer_fc3/dense/kernel*
dtype0*
_output_shapes
:
Ł
3layer_fc3/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc3/dense/kernel*
dtype0*
_output_shapes
: 

-layer_fc3/dense/kernel/Adam/Initializer/zerosFill=layer_fc3/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3layer_fc3/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc3/dense/kernel* 
_output_shapes
:

ž
layer_fc3/dense/kernel/Adam
VariableV2*
shared_name *)
_class
loc:@layer_fc3/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

÷
"layer_fc3/dense/kernel/Adam/AssignAssignlayer_fc3/dense/kernel/Adam-layer_fc3/dense/kernel/Adam/Initializer/zeros*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel

 layer_fc3/dense/kernel/Adam/readIdentitylayer_fc3/dense/kernel/Adam*)
_class
loc:@layer_fc3/dense/kernel* 
_output_shapes
:
*
T0
ť
?layer_fc3/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@layer_fc3/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc3/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc3/dense/kernel*
dtype0*
_output_shapes
: 

/layer_fc3/dense/kernel/Adam_1/Initializer/zerosFill?layer_fc3/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5layer_fc3/dense/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*

index_type0*)
_class
loc:@layer_fc3/dense/kernel
Ŕ
layer_fc3/dense/kernel/Adam_1
VariableV2*
shared_name *)
_class
loc:@layer_fc3/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:

ý
$layer_fc3/dense/kernel/Adam_1/AssignAssignlayer_fc3/dense/kernel/Adam_1/layer_fc3/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:

Ł
"layer_fc3/dense/kernel/Adam_1/readIdentitylayer_fc3/dense/kernel/Adam_1*
T0*)
_class
loc:@layer_fc3/dense/kernel* 
_output_shapes
:

Ł
+layer_fc3/dense/bias/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc3/dense/bias*
dtype0*
_output_shapes	
:
°
layer_fc3/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@layer_fc3/dense/bias*
	container *
shape:
ę
 layer_fc3/dense/bias/Adam/AssignAssignlayer_fc3/dense/bias/Adam+layer_fc3/dense/bias/Adam/Initializer/zeros*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(

layer_fc3/dense/bias/Adam/readIdentitylayer_fc3/dense/bias/Adam*
T0*'
_class
loc:@layer_fc3/dense/bias*
_output_shapes	
:
Ľ
-layer_fc3/dense/bias/Adam_1/Initializer/zerosConst*
_output_shapes	
:*
valueB*    *'
_class
loc:@layer_fc3/dense/bias*
dtype0
˛
layer_fc3/dense/bias/Adam_1
VariableV2*
_output_shapes	
:*
shared_name *'
_class
loc:@layer_fc3/dense/bias*
	container *
shape:*
dtype0
đ
"layer_fc3/dense/bias/Adam_1/AssignAssignlayer_fc3/dense/bias/Adam_1-layer_fc3/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias

 layer_fc3/dense/bias/Adam_1/readIdentitylayer_fc3/dense/bias/Adam_1*
T0*'
_class
loc:@layer_fc3/dense/bias*
_output_shapes	
:
š
=layer_fc4/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
:
Ł
3layer_fc4/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
: 

-layer_fc4/dense/kernel/Adam/Initializer/zerosFill=layer_fc4/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3layer_fc4/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc4/dense/kernel* 
_output_shapes
:

ž
layer_fc4/dense/kernel/Adam
VariableV2*)
_class
loc:@layer_fc4/dense/kernel*
	container *
shape:
*
dtype0* 
_output_shapes
:
*
shared_name 
÷
"layer_fc4/dense/kernel/Adam/AssignAssignlayer_fc4/dense/kernel/Adam-layer_fc4/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:


 layer_fc4/dense/kernel/Adam/readIdentitylayer_fc4/dense/kernel/Adam*
T0*)
_class
loc:@layer_fc4/dense/kernel* 
_output_shapes
:

ť
?layer_fc4/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc4/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc4/dense/kernel*
dtype0*
_output_shapes
: 

/layer_fc4/dense/kernel/Adam_1/Initializer/zerosFill?layer_fc4/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5layer_fc4/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc4/dense/kernel* 
_output_shapes
:

Ŕ
layer_fc4/dense/kernel/Adam_1
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@layer_fc4/dense/kernel*
	container *
shape:

ý
$layer_fc4/dense/kernel/Adam_1/AssignAssignlayer_fc4/dense/kernel/Adam_1/layer_fc4/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:

Ł
"layer_fc4/dense/kernel/Adam_1/readIdentitylayer_fc4/dense/kernel/Adam_1*)
_class
loc:@layer_fc4/dense/kernel* 
_output_shapes
:
*
T0
Ł
+layer_fc4/dense/bias/Adam/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *'
_class
loc:@layer_fc4/dense/bias
°
layer_fc4/dense/bias/Adam
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@layer_fc4/dense/bias*
	container *
shape:
ę
 layer_fc4/dense/bias/Adam/AssignAssignlayer_fc4/dense/bias/Adam+layer_fc4/dense/bias/Adam/Initializer/zeros*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias

layer_fc4/dense/bias/Adam/readIdentitylayer_fc4/dense/bias/Adam*
_output_shapes	
:*
T0*'
_class
loc:@layer_fc4/dense/bias
Ľ
-layer_fc4/dense/bias/Adam_1/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc4/dense/bias*
dtype0*
_output_shapes	
:
˛
layer_fc4/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@layer_fc4/dense/bias*
	container *
shape:
đ
"layer_fc4/dense/bias/Adam_1/AssignAssignlayer_fc4/dense/bias/Adam_1-layer_fc4/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:

 layer_fc4/dense/bias/Adam_1/readIdentitylayer_fc4/dense/bias/Adam_1*
_output_shapes	
:*
T0*'
_class
loc:@layer_fc4/dense/bias
š
=layer_fc5/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@layer_fc5/dense/kernel*
dtype0*
_output_shapes
:
Ł
3layer_fc5/dense/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc5/dense/kernel*
dtype0*
_output_shapes
: 

-layer_fc5/dense/kernel/Adam/Initializer/zerosFill=layer_fc5/dense/kernel/Adam/Initializer/zeros/shape_as_tensor3layer_fc5/dense/kernel/Adam/Initializer/zeros/Const*
T0*

index_type0*)
_class
loc:@layer_fc5/dense/kernel* 
_output_shapes
:

ž
layer_fc5/dense/kernel/Adam
VariableV2*
dtype0* 
_output_shapes
:
*
shared_name *)
_class
loc:@layer_fc5/dense/kernel*
	container *
shape:

÷
"layer_fc5/dense/kernel/Adam/AssignAssignlayer_fc5/dense/kernel/Adam-layer_fc5/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:


 layer_fc5/dense/kernel/Adam/readIdentitylayer_fc5/dense/kernel/Adam* 
_output_shapes
:
*
T0*)
_class
loc:@layer_fc5/dense/kernel
ť
?layer_fc5/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
valueB"      *)
_class
loc:@layer_fc5/dense/kernel*
dtype0*
_output_shapes
:
Ľ
5layer_fc5/dense/kernel/Adam_1/Initializer/zeros/ConstConst*
valueB
 *    *)
_class
loc:@layer_fc5/dense/kernel*
dtype0*
_output_shapes
: 

/layer_fc5/dense/kernel/Adam_1/Initializer/zerosFill?layer_fc5/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor5layer_fc5/dense/kernel/Adam_1/Initializer/zeros/Const* 
_output_shapes
:
*
T0*

index_type0*)
_class
loc:@layer_fc5/dense/kernel
Ŕ
layer_fc5/dense/kernel/Adam_1
VariableV2* 
_output_shapes
:
*
shared_name *)
_class
loc:@layer_fc5/dense/kernel*
	container *
shape:
*
dtype0
ý
$layer_fc5/dense/kernel/Adam_1/AssignAssignlayer_fc5/dense/kernel/Adam_1/layer_fc5/dense/kernel/Adam_1/Initializer/zeros*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ł
"layer_fc5/dense/kernel/Adam_1/readIdentitylayer_fc5/dense/kernel/Adam_1*
T0*)
_class
loc:@layer_fc5/dense/kernel* 
_output_shapes
:

Ł
+layer_fc5/dense/bias/Adam/Initializer/zerosConst*
valueB*    *'
_class
loc:@layer_fc5/dense/bias*
dtype0*
_output_shapes	
:
°
layer_fc5/dense/bias/Adam
VariableV2*'
_class
loc:@layer_fc5/dense/bias*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name 
ę
 layer_fc5/dense/bias/Adam/AssignAssignlayer_fc5/dense/bias/Adam+layer_fc5/dense/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:

layer_fc5/dense/bias/Adam/readIdentitylayer_fc5/dense/bias/Adam*
T0*'
_class
loc:@layer_fc5/dense/bias*
_output_shapes	
:
Ľ
-layer_fc5/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes	
:*
valueB*    *'
_class
loc:@layer_fc5/dense/bias
˛
layer_fc5/dense/bias/Adam_1
VariableV2*
	container *
shape:*
dtype0*
_output_shapes	
:*
shared_name *'
_class
loc:@layer_fc5/dense/bias
đ
"layer_fc5/dense/bias/Adam_1/AssignAssignlayer_fc5/dense/bias/Adam_1-layer_fc5/dense/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:

 layer_fc5/dense/bias/Adam_1/readIdentitylayer_fc5/dense/bias/Adam_1*
_output_shapes	
:*
T0*'
_class
loc:@layer_fc5/dense/bias

#dense/kernel/Adam/Initializer/zerosConst*
dtype0*
_output_shapes
:	*
valueB	*    *
_class
loc:@dense/kernel
¨
dense/kernel/Adam
VariableV2*
	container *
shape:	*
dtype0*
_output_shapes
:	*
shared_name *
_class
loc:@dense/kernel
Î
dense/kernel/Adam/AssignAssigndense/kernel/Adam#dense/kernel/Adam/Initializer/zeros*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@dense/kernel

dense/kernel/Adam/readIdentitydense/kernel/Adam*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	

%dense/kernel/Adam_1/Initializer/zerosConst*
valueB	*    *
_class
loc:@dense/kernel*
dtype0*
_output_shapes
:	
Ş
dense/kernel/Adam_1
VariableV2*
shared_name *
_class
loc:@dense/kernel*
	container *
shape:	*
dtype0*
_output_shapes
:	
Ô
dense/kernel/Adam_1/AssignAssigndense/kernel/Adam_1%dense/kernel/Adam_1/Initializer/zeros*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(

dense/kernel/Adam_1/readIdentitydense/kernel/Adam_1*
T0*
_class
loc:@dense/kernel*
_output_shapes
:	
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
dtype0*
_output_shapes
: *
valueB
 *wž?
Q
Adam/epsilonConst*
valueB
 *ÍĚĚ=*
dtype0*
_output_shapes
: 
ń
,Adam/update_layer_fc0/dense/kernel/ApplyAdam	ApplyAdamlayer_fc0/dense/kernellayer_fc0/dense/kernel/Adamlayer_fc0/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_6*)
_class
loc:@layer_fc0/dense/kernel*
use_nesterov( *
_output_shapes

: *
use_locking( *
T0

*Adam/update_layer_fc0/dense/bias/ApplyAdam	ApplyAdamlayer_fc0/dense/biaslayer_fc0/dense/bias/Adamlayer_fc0/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonAgradients/layer_fc0/dense/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
: *
use_locking( *
T0*'
_class
loc:@layer_fc0/dense/bias
ń
,Adam/update_layer_fc1/dense/kernel/ApplyAdam	ApplyAdamlayer_fc1/dense/kernellayer_fc1/dense/kernel/Adamlayer_fc1/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_5*
T0*)
_class
loc:@layer_fc1/dense/kernel*
use_nesterov( *
_output_shapes

: @*
use_locking( 

*Adam/update_layer_fc1/dense/bias/ApplyAdam	ApplyAdamlayer_fc1/dense/biaslayer_fc1/dense/bias/Adamlayer_fc1/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonAgradients/layer_fc1/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@layer_fc1/dense/bias*
use_nesterov( *
_output_shapes
:@
ň
,Adam/update_layer_fc2/dense/kernel/ApplyAdam	ApplyAdamlayer_fc2/dense/kernellayer_fc2/dense/kernel/Adamlayer_fc2/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_4*
_output_shapes
:	@*
use_locking( *
T0*)
_class
loc:@layer_fc2/dense/kernel*
use_nesterov( 

*Adam/update_layer_fc2/dense/bias/ApplyAdam	ApplyAdamlayer_fc2/dense/biaslayer_fc2/dense/bias/Adamlayer_fc2/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonAgradients/layer_fc2/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*'
_class
loc:@layer_fc2/dense/bias*
use_nesterov( 
ó
,Adam/update_layer_fc3/dense/kernel/ApplyAdam	ApplyAdamlayer_fc3/dense/kernellayer_fc3/dense/kernel/Adamlayer_fc3/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_3*
use_locking( *
T0*)
_class
loc:@layer_fc3/dense/kernel*
use_nesterov( * 
_output_shapes
:


*Adam/update_layer_fc3/dense/bias/ApplyAdam	ApplyAdamlayer_fc3/dense/biaslayer_fc3/dense/bias/Adamlayer_fc3/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonAgradients/layer_fc3/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*'
_class
loc:@layer_fc3/dense/bias*
use_nesterov( 
ó
,Adam/update_layer_fc4/dense/kernel/ApplyAdam	ApplyAdamlayer_fc4/dense/kernellayer_fc4/dense/kernel/Adamlayer_fc4/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_2*
use_locking( *
T0*)
_class
loc:@layer_fc4/dense/kernel*
use_nesterov( * 
_output_shapes
:


*Adam/update_layer_fc4/dense/bias/ApplyAdam	ApplyAdamlayer_fc4/dense/biaslayer_fc4/dense/bias/Adamlayer_fc4/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonAgradients/layer_fc4/dense/BiasAdd_grad/tuple/control_dependency_1*
T0*'
_class
loc:@layer_fc4/dense/bias*
use_nesterov( *
_output_shapes	
:*
use_locking( 
ó
,Adam/update_layer_fc5/dense/kernel/ApplyAdam	ApplyAdamlayer_fc5/dense/kernellayer_fc5/dense/kernel/Adamlayer_fc5/dense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilongradients/AddN_1*
use_locking( *
T0*)
_class
loc:@layer_fc5/dense/kernel*
use_nesterov( * 
_output_shapes
:


*Adam/update_layer_fc5/dense/bias/ApplyAdam	ApplyAdamlayer_fc5/dense/biaslayer_fc5/dense/bias/Adamlayer_fc5/dense/bias/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilonAgradients/layer_fc5/dense/BiasAdd_grad/tuple/control_dependency_1*
_output_shapes	
:*
use_locking( *
T0*'
_class
loc:@layer_fc5/dense/bias*
use_nesterov( 
ć
"Adam/update_dense/kernel/ApplyAdam	ApplyAdamdense/kerneldense/kernel/Adamdense/kernel/Adam_1beta1_power/readbeta2_power/readPlaceholder
Adam/beta1
Adam/beta2Adam/epsilon6gradients/dense/MatMul_grad/tuple/control_dependency_1*
_output_shapes
:	*
use_locking( *
T0*
_class
loc:@dense/kernel*
use_nesterov( 
ź
Adam/mulMulbeta1_power/read
Adam/beta1#^Adam/update_dense/kernel/ApplyAdam+^Adam/update_layer_fc0/dense/bias/ApplyAdam-^Adam/update_layer_fc0/dense/kernel/ApplyAdam+^Adam/update_layer_fc1/dense/bias/ApplyAdam-^Adam/update_layer_fc1/dense/kernel/ApplyAdam+^Adam/update_layer_fc2/dense/bias/ApplyAdam-^Adam/update_layer_fc2/dense/kernel/ApplyAdam+^Adam/update_layer_fc3/dense/bias/ApplyAdam-^Adam/update_layer_fc3/dense/kernel/ApplyAdam+^Adam/update_layer_fc4/dense/bias/ApplyAdam-^Adam/update_layer_fc4/dense/kernel/ApplyAdam+^Adam/update_layer_fc5/dense/bias/ApplyAdam-^Adam/update_layer_fc5/dense/kernel/ApplyAdam*
_class
loc:@dense/kernel*
_output_shapes
: *
T0

Adam/AssignAssignbeta1_powerAdam/mul*
use_locking( *
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: 
ž

Adam/mul_1Mulbeta2_power/read
Adam/beta2#^Adam/update_dense/kernel/ApplyAdam+^Adam/update_layer_fc0/dense/bias/ApplyAdam-^Adam/update_layer_fc0/dense/kernel/ApplyAdam+^Adam/update_layer_fc1/dense/bias/ApplyAdam-^Adam/update_layer_fc1/dense/kernel/ApplyAdam+^Adam/update_layer_fc2/dense/bias/ApplyAdam-^Adam/update_layer_fc2/dense/kernel/ApplyAdam+^Adam/update_layer_fc3/dense/bias/ApplyAdam-^Adam/update_layer_fc3/dense/kernel/ApplyAdam+^Adam/update_layer_fc4/dense/bias/ApplyAdam-^Adam/update_layer_fc4/dense/kernel/ApplyAdam+^Adam/update_layer_fc5/dense/bias/ApplyAdam-^Adam/update_layer_fc5/dense/kernel/ApplyAdam*
T0*
_class
loc:@dense/kernel*
_output_shapes
: 

Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: *
use_locking( 
ţ
Adam/updateNoOp^Adam/Assign^Adam/Assign_1#^Adam/update_dense/kernel/ApplyAdam+^Adam/update_layer_fc0/dense/bias/ApplyAdam-^Adam/update_layer_fc0/dense/kernel/ApplyAdam+^Adam/update_layer_fc1/dense/bias/ApplyAdam-^Adam/update_layer_fc1/dense/kernel/ApplyAdam+^Adam/update_layer_fc2/dense/bias/ApplyAdam-^Adam/update_layer_fc2/dense/kernel/ApplyAdam+^Adam/update_layer_fc3/dense/bias/ApplyAdam-^Adam/update_layer_fc3/dense/kernel/ApplyAdam+^Adam/update_layer_fc4/dense/bias/ApplyAdam-^Adam/update_layer_fc4/dense/kernel/ApplyAdam+^Adam/update_layer_fc5/dense/bias/ApplyAdam-^Adam/update_layer_fc5/dense/kernel/ApplyAdam
z

Adam/valueConst^Adam/update*
value	B :*
_class
loc:@global_step*
dtype0*
_output_shapes
: 
~
Adam	AssignAddglobal_step
Adam/value*
use_locking( *
T0*
_class
loc:@global_step*
_output_shapes
: 
P

save/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
	
save/SaveV2/tensor_namesConst*ž
value´Bą*Bbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBlayer_fc0/dense/biasBlayer_fc0/dense/bias/AdamBlayer_fc0/dense/bias/Adam_1Blayer_fc0/dense/kernelBlayer_fc0/dense/kernel/AdamBlayer_fc0/dense/kernel/Adam_1Blayer_fc1/dense/biasBlayer_fc1/dense/bias/AdamBlayer_fc1/dense/bias/Adam_1Blayer_fc1/dense/kernelBlayer_fc1/dense/kernel/AdamBlayer_fc1/dense/kernel/Adam_1Blayer_fc2/dense/biasBlayer_fc2/dense/bias/AdamBlayer_fc2/dense/bias/Adam_1Blayer_fc2/dense/kernelBlayer_fc2/dense/kernel/AdamBlayer_fc2/dense/kernel/Adam_1Blayer_fc3/dense/biasBlayer_fc3/dense/bias/AdamBlayer_fc3/dense/bias/Adam_1Blayer_fc3/dense/kernelBlayer_fc3/dense/kernel/AdamBlayer_fc3/dense/kernel/Adam_1Blayer_fc4/dense/biasBlayer_fc4/dense/bias/AdamBlayer_fc4/dense/bias/Adam_1Blayer_fc4/dense/kernelBlayer_fc4/dense/kernel/AdamBlayer_fc4/dense/kernel/Adam_1Blayer_fc5/dense/biasBlayer_fc5/dense/bias/AdamBlayer_fc5/dense/bias/Adam_1Blayer_fc5/dense/kernelBlayer_fc5/dense/kernel/AdamBlayer_fc5/dense/kernel/Adam_1*
dtype0*
_output_shapes
:*
ˇ
save/SaveV2/shape_and_slicesConst*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:*
ź	
save/SaveV2SaveV2
save/Constsave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerdense/kerneldense/kernel/Adamdense/kernel/Adam_1global_steplayer_fc0/dense/biaslayer_fc0/dense/bias/Adamlayer_fc0/dense/bias/Adam_1layer_fc0/dense/kernellayer_fc0/dense/kernel/Adamlayer_fc0/dense/kernel/Adam_1layer_fc1/dense/biaslayer_fc1/dense/bias/Adamlayer_fc1/dense/bias/Adam_1layer_fc1/dense/kernellayer_fc1/dense/kernel/Adamlayer_fc1/dense/kernel/Adam_1layer_fc2/dense/biaslayer_fc2/dense/bias/Adamlayer_fc2/dense/bias/Adam_1layer_fc2/dense/kernellayer_fc2/dense/kernel/Adamlayer_fc2/dense/kernel/Adam_1layer_fc3/dense/biaslayer_fc3/dense/bias/Adamlayer_fc3/dense/bias/Adam_1layer_fc3/dense/kernellayer_fc3/dense/kernel/Adamlayer_fc3/dense/kernel/Adam_1layer_fc4/dense/biaslayer_fc4/dense/bias/Adamlayer_fc4/dense/bias/Adam_1layer_fc4/dense/kernellayer_fc4/dense/kernel/Adamlayer_fc4/dense/kernel/Adam_1layer_fc5/dense/biaslayer_fc5/dense/bias/Adamlayer_fc5/dense/bias/Adam_1layer_fc5/dense/kernellayer_fc5/dense/kernel/Adamlayer_fc5/dense/kernel/Adam_1*8
dtypes.
,2*
}
save/control_dependencyIdentity
save/Const^save/SaveV2*
T0*
_class
loc:@save/Const*
_output_shapes
: 
	
save/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:**ž
value´Bą*Bbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBlayer_fc0/dense/biasBlayer_fc0/dense/bias/AdamBlayer_fc0/dense/bias/Adam_1Blayer_fc0/dense/kernelBlayer_fc0/dense/kernel/AdamBlayer_fc0/dense/kernel/Adam_1Blayer_fc1/dense/biasBlayer_fc1/dense/bias/AdamBlayer_fc1/dense/bias/Adam_1Blayer_fc1/dense/kernelBlayer_fc1/dense/kernel/AdamBlayer_fc1/dense/kernel/Adam_1Blayer_fc2/dense/biasBlayer_fc2/dense/bias/AdamBlayer_fc2/dense/bias/Adam_1Blayer_fc2/dense/kernelBlayer_fc2/dense/kernel/AdamBlayer_fc2/dense/kernel/Adam_1Blayer_fc3/dense/biasBlayer_fc3/dense/bias/AdamBlayer_fc3/dense/bias/Adam_1Blayer_fc3/dense/kernelBlayer_fc3/dense/kernel/AdamBlayer_fc3/dense/kernel/Adam_1Blayer_fc4/dense/biasBlayer_fc4/dense/bias/AdamBlayer_fc4/dense/bias/Adam_1Blayer_fc4/dense/kernelBlayer_fc4/dense/kernel/AdamBlayer_fc4/dense/kernel/Adam_1Blayer_fc5/dense/biasBlayer_fc5/dense/bias/AdamBlayer_fc5/dense/bias/Adam_1Blayer_fc5/dense/kernelBlayer_fc5/dense/kernel/AdamBlayer_fc5/dense/kernel/Adam_1
ş
save/RestoreV2/shape_and_slicesConst*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:*
ŕ
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*8
dtypes.
,2**ž
_output_shapesŤ
¨::::::::::::::::::::::::::::::::::::::::::

save/AssignAssignbeta1_powersave/RestoreV2*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: 
Ą
save/Assign_1Assignbeta2_powersave/RestoreV2:1*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@dense/kernel
Ť
save/Assign_2Assigndense/kernelsave/RestoreV2:2*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
°
save/Assign_3Assigndense/kernel/Adamsave/RestoreV2:3*
_output_shapes
:	*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(
˛
save/Assign_4Assigndense/kernel/Adam_1save/RestoreV2:4*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(
 
save/Assign_5Assignglobal_stepsave/RestoreV2:5*
T0*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: *
use_locking(
ś
save/Assign_6Assignlayer_fc0/dense/biassave/RestoreV2:6*
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(*
_output_shapes
: 
ť
save/Assign_7Assignlayer_fc0/dense/bias/Adamsave/RestoreV2:7*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(
˝
save/Assign_8Assignlayer_fc0/dense/bias/Adam_1save/RestoreV2:8*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(
ž
save/Assign_9Assignlayer_fc0/dense/kernelsave/RestoreV2:9*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 
Ĺ
save/Assign_10Assignlayer_fc0/dense/kernel/Adamsave/RestoreV2:10*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
Ç
save/Assign_11Assignlayer_fc0/dense/kernel/Adam_1save/RestoreV2:11*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: *
use_locking(*
T0
¸
save/Assign_12Assignlayer_fc1/dense/biassave/RestoreV2:12*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@
˝
save/Assign_13Assignlayer_fc1/dense/bias/Adamsave/RestoreV2:13*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@
ż
save/Assign_14Assignlayer_fc1/dense/bias/Adam_1save/RestoreV2:14*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@
Ŕ
save/Assign_15Assignlayer_fc1/dense/kernelsave/RestoreV2:15*
validate_shape(*
_output_shapes

: @*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel
Ĺ
save/Assign_16Assignlayer_fc1/dense/kernel/Adamsave/RestoreV2:16*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @
Ç
save/Assign_17Assignlayer_fc1/dense/kernel/Adam_1save/RestoreV2:17*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @
š
save/Assign_18Assignlayer_fc2/dense/biassave/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_19Assignlayer_fc2/dense/bias/Adamsave/RestoreV2:19*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ŕ
save/Assign_20Assignlayer_fc2/dense/bias/Adam_1save/RestoreV2:20*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:
Á
save/Assign_21Assignlayer_fc2/dense/kernelsave/RestoreV2:21*
use_locking(*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@
Ć
save/Assign_22Assignlayer_fc2/dense/kernel/Adamsave/RestoreV2:22*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(*
T0
Č
save/Assign_23Assignlayer_fc2/dense/kernel/Adam_1save/RestoreV2:23*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(
š
save/Assign_24Assignlayer_fc3/dense/biassave/RestoreV2:24*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_25Assignlayer_fc3/dense/bias/Adamsave/RestoreV2:25*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ŕ
save/Assign_26Assignlayer_fc3/dense/bias/Adam_1save/RestoreV2:26*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Â
save/Assign_27Assignlayer_fc3/dense/kernelsave/RestoreV2:27*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ç
save/Assign_28Assignlayer_fc3/dense/kernel/Adamsave/RestoreV2:28*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:

É
save/Assign_29Assignlayer_fc3/dense/kernel/Adam_1save/RestoreV2:29*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel
š
save/Assign_30Assignlayer_fc4/dense/biassave/RestoreV2:30*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_31Assignlayer_fc4/dense/bias/Adamsave/RestoreV2:31*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Ŕ
save/Assign_32Assignlayer_fc4/dense/bias/Adam_1save/RestoreV2:32*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:
Â
save/Assign_33Assignlayer_fc4/dense/kernelsave/RestoreV2:33*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:

Ç
save/Assign_34Assignlayer_fc4/dense/kernel/Adamsave/RestoreV2:34*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:

É
save/Assign_35Assignlayer_fc4/dense/kernel/Adam_1save/RestoreV2:35*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
š
save/Assign_36Assignlayer_fc5/dense/biassave/RestoreV2:36*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:
ž
save/Assign_37Assignlayer_fc5/dense/bias/Adamsave/RestoreV2:37*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(
Ŕ
save/Assign_38Assignlayer_fc5/dense/bias/Adam_1save/RestoreV2:38*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(
Â
save/Assign_39Assignlayer_fc5/dense/kernelsave/RestoreV2:39*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
Ç
save/Assign_40Assignlayer_fc5/dense/kernel/Adamsave/RestoreV2:40*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:

É
save/Assign_41Assignlayer_fc5/dense/kernel/Adam_1save/RestoreV2:41*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:

Ö
save/restore_allNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9

initNoOp^beta1_power/Assign^beta2_power/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^global_step/Assign!^layer_fc0/dense/bias/Adam/Assign#^layer_fc0/dense/bias/Adam_1/Assign^layer_fc0/dense/bias/Assign#^layer_fc0/dense/kernel/Adam/Assign%^layer_fc0/dense/kernel/Adam_1/Assign^layer_fc0/dense/kernel/Assign!^layer_fc1/dense/bias/Adam/Assign#^layer_fc1/dense/bias/Adam_1/Assign^layer_fc1/dense/bias/Assign#^layer_fc1/dense/kernel/Adam/Assign%^layer_fc1/dense/kernel/Adam_1/Assign^layer_fc1/dense/kernel/Assign!^layer_fc2/dense/bias/Adam/Assign#^layer_fc2/dense/bias/Adam_1/Assign^layer_fc2/dense/bias/Assign#^layer_fc2/dense/kernel/Adam/Assign%^layer_fc2/dense/kernel/Adam_1/Assign^layer_fc2/dense/kernel/Assign!^layer_fc3/dense/bias/Adam/Assign#^layer_fc3/dense/bias/Adam_1/Assign^layer_fc3/dense/bias/Assign#^layer_fc3/dense/kernel/Adam/Assign%^layer_fc3/dense/kernel/Adam_1/Assign^layer_fc3/dense/kernel/Assign!^layer_fc4/dense/bias/Adam/Assign#^layer_fc4/dense/bias/Adam_1/Assign^layer_fc4/dense/bias/Assign#^layer_fc4/dense/kernel/Adam/Assign%^layer_fc4/dense/kernel/Adam_1/Assign^layer_fc4/dense/kernel/Assign!^layer_fc5/dense/bias/Adam/Assign#^layer_fc5/dense/bias/Adam_1/Assign^layer_fc5/dense/bias/Assign#^layer_fc5/dense/kernel/Adam/Assign%^layer_fc5/dense/kernel/Adam_1/Assign^layer_fc5/dense/kernel/Assign
N
	loss/tagsConst*
_output_shapes
: *
valueB
 Bloss*
dtype0
H
lossScalarSummary	loss/tagsadd_2*
_output_shapes
: *
T0
`
learning_rate/tagsConst*
valueB Blearning_rate*
dtype0*
_output_shapes
: 
`
learning_rateScalarSummarylearning_rate/tagsPlaceholder*
T0*
_output_shapes
: 
a
histogram_loss/tagConst*
valueB Bhistogram_loss*
dtype0*
_output_shapes
: 
^
histogram_lossHistogramSummaryhistogram_loss/tagadd_2*
T0*
_output_shapes
: 
h
Merge/MergeSummaryMergeSummarylosslearning_ratehistogram_loss*
N*
_output_shapes
: 
R
save_1/ConstConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
	
save_1/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:**ž
value´Bą*Bbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBlayer_fc0/dense/biasBlayer_fc0/dense/bias/AdamBlayer_fc0/dense/bias/Adam_1Blayer_fc0/dense/kernelBlayer_fc0/dense/kernel/AdamBlayer_fc0/dense/kernel/Adam_1Blayer_fc1/dense/biasBlayer_fc1/dense/bias/AdamBlayer_fc1/dense/bias/Adam_1Blayer_fc1/dense/kernelBlayer_fc1/dense/kernel/AdamBlayer_fc1/dense/kernel/Adam_1Blayer_fc2/dense/biasBlayer_fc2/dense/bias/AdamBlayer_fc2/dense/bias/Adam_1Blayer_fc2/dense/kernelBlayer_fc2/dense/kernel/AdamBlayer_fc2/dense/kernel/Adam_1Blayer_fc3/dense/biasBlayer_fc3/dense/bias/AdamBlayer_fc3/dense/bias/Adam_1Blayer_fc3/dense/kernelBlayer_fc3/dense/kernel/AdamBlayer_fc3/dense/kernel/Adam_1Blayer_fc4/dense/biasBlayer_fc4/dense/bias/AdamBlayer_fc4/dense/bias/Adam_1Blayer_fc4/dense/kernelBlayer_fc4/dense/kernel/AdamBlayer_fc4/dense/kernel/Adam_1Blayer_fc5/dense/biasBlayer_fc5/dense/bias/AdamBlayer_fc5/dense/bias/Adam_1Blayer_fc5/dense/kernelBlayer_fc5/dense/kernel/AdamBlayer_fc5/dense/kernel/Adam_1
š
save_1/SaveV2/shape_and_slicesConst*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:*
Ä	
save_1/SaveV2SaveV2save_1/Constsave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_powerdense/kerneldense/kernel/Adamdense/kernel/Adam_1global_steplayer_fc0/dense/biaslayer_fc0/dense/bias/Adamlayer_fc0/dense/bias/Adam_1layer_fc0/dense/kernellayer_fc0/dense/kernel/Adamlayer_fc0/dense/kernel/Adam_1layer_fc1/dense/biaslayer_fc1/dense/bias/Adamlayer_fc1/dense/bias/Adam_1layer_fc1/dense/kernellayer_fc1/dense/kernel/Adamlayer_fc1/dense/kernel/Adam_1layer_fc2/dense/biaslayer_fc2/dense/bias/Adamlayer_fc2/dense/bias/Adam_1layer_fc2/dense/kernellayer_fc2/dense/kernel/Adamlayer_fc2/dense/kernel/Adam_1layer_fc3/dense/biaslayer_fc3/dense/bias/Adamlayer_fc3/dense/bias/Adam_1layer_fc3/dense/kernellayer_fc3/dense/kernel/Adamlayer_fc3/dense/kernel/Adam_1layer_fc4/dense/biaslayer_fc4/dense/bias/Adamlayer_fc4/dense/bias/Adam_1layer_fc4/dense/kernellayer_fc4/dense/kernel/Adamlayer_fc4/dense/kernel/Adam_1layer_fc5/dense/biaslayer_fc5/dense/bias/Adamlayer_fc5/dense/bias/Adam_1layer_fc5/dense/kernellayer_fc5/dense/kernel/Adamlayer_fc5/dense/kernel/Adam_1*8
dtypes.
,2*

save_1/control_dependencyIdentitysave_1/Const^save_1/SaveV2*
_class
loc:@save_1/Const*
_output_shapes
: *
T0
	
save_1/RestoreV2/tensor_namesConst*ž
value´Bą*Bbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBlayer_fc0/dense/biasBlayer_fc0/dense/bias/AdamBlayer_fc0/dense/bias/Adam_1Blayer_fc0/dense/kernelBlayer_fc0/dense/kernel/AdamBlayer_fc0/dense/kernel/Adam_1Blayer_fc1/dense/biasBlayer_fc1/dense/bias/AdamBlayer_fc1/dense/bias/Adam_1Blayer_fc1/dense/kernelBlayer_fc1/dense/kernel/AdamBlayer_fc1/dense/kernel/Adam_1Blayer_fc2/dense/biasBlayer_fc2/dense/bias/AdamBlayer_fc2/dense/bias/Adam_1Blayer_fc2/dense/kernelBlayer_fc2/dense/kernel/AdamBlayer_fc2/dense/kernel/Adam_1Blayer_fc3/dense/biasBlayer_fc3/dense/bias/AdamBlayer_fc3/dense/bias/Adam_1Blayer_fc3/dense/kernelBlayer_fc3/dense/kernel/AdamBlayer_fc3/dense/kernel/Adam_1Blayer_fc4/dense/biasBlayer_fc4/dense/bias/AdamBlayer_fc4/dense/bias/Adam_1Blayer_fc4/dense/kernelBlayer_fc4/dense/kernel/AdamBlayer_fc4/dense/kernel/Adam_1Blayer_fc5/dense/biasBlayer_fc5/dense/bias/AdamBlayer_fc5/dense/bias/Adam_1Blayer_fc5/dense/kernelBlayer_fc5/dense/kernel/AdamBlayer_fc5/dense/kernel/Adam_1*
dtype0*
_output_shapes
:*
ź
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:**g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
č
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*8
dtypes.
,2**ž
_output_shapesŤ
¨::::::::::::::::::::::::::::::::::::::::::
Ą
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
Ľ
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: *
use_locking(
Ż
save_1/Assign_2Assigndense/kernelsave_1/RestoreV2:2*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	*
use_locking(*
T0
´
save_1/Assign_3Assigndense/kernel/Adamsave_1/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	
ś
save_1/Assign_4Assigndense/kernel/Adam_1save_1/RestoreV2:4*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	
¤
save_1/Assign_5Assignglobal_stepsave_1/RestoreV2:5*
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(*
_output_shapes
: 
ş
save_1/Assign_6Assignlayer_fc0/dense/biassave_1/RestoreV2:6*
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(*
_output_shapes
: 
ż
save_1/Assign_7Assignlayer_fc0/dense/bias/Adamsave_1/RestoreV2:7*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
Á
save_1/Assign_8Assignlayer_fc0/dense/bias/Adam_1save_1/RestoreV2:8*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(
Â
save_1/Assign_9Assignlayer_fc0/dense/kernelsave_1/RestoreV2:9*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 
É
save_1/Assign_10Assignlayer_fc0/dense/kernel/Adamsave_1/RestoreV2:10*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 
Ë
save_1/Assign_11Assignlayer_fc0/dense/kernel/Adam_1save_1/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 
ź
save_1/Assign_12Assignlayer_fc1/dense/biassave_1/RestoreV2:12*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@
Á
save_1/Assign_13Assignlayer_fc1/dense/bias/Adamsave_1/RestoreV2:13*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@
Ă
save_1/Assign_14Assignlayer_fc1/dense/bias/Adam_1save_1/RestoreV2:14*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(
Ä
save_1/Assign_15Assignlayer_fc1/dense/kernelsave_1/RestoreV2:15*
validate_shape(*
_output_shapes

: @*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel
É
save_1/Assign_16Assignlayer_fc1/dense/kernel/Adamsave_1/RestoreV2:16*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @*
use_locking(
Ë
save_1/Assign_17Assignlayer_fc1/dense/kernel/Adam_1save_1/RestoreV2:17*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @
˝
save_1/Assign_18Assignlayer_fc2/dense/biassave_1/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:
Â
save_1/Assign_19Assignlayer_fc2/dense/bias/Adamsave_1/RestoreV2:19*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(
Ä
save_1/Assign_20Assignlayer_fc2/dense/bias/Adam_1save_1/RestoreV2:20*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:
Ĺ
save_1/Assign_21Assignlayer_fc2/dense/kernelsave_1/RestoreV2:21*
use_locking(*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@
Ę
save_1/Assign_22Assignlayer_fc2/dense/kernel/Adamsave_1/RestoreV2:22*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(
Ě
save_1/Assign_23Assignlayer_fc2/dense/kernel/Adam_1save_1/RestoreV2:23*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(
˝
save_1/Assign_24Assignlayer_fc3/dense/biassave_1/RestoreV2:24*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(*
_output_shapes	
:
Â
save_1/Assign_25Assignlayer_fc3/dense/bias/Adamsave_1/RestoreV2:25*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(*
_output_shapes	
:
Ä
save_1/Assign_26Assignlayer_fc3/dense/bias/Adam_1save_1/RestoreV2:26*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(*
_output_shapes	
:
Ć
save_1/Assign_27Assignlayer_fc3/dense/kernelsave_1/RestoreV2:27*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:

Ë
save_1/Assign_28Assignlayer_fc3/dense/kernel/Adamsave_1/RestoreV2:28*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:

Í
save_1/Assign_29Assignlayer_fc3/dense/kernel/Adam_1save_1/RestoreV2:29*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
˝
save_1/Assign_30Assignlayer_fc4/dense/biassave_1/RestoreV2:30*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias
Â
save_1/Assign_31Assignlayer_fc4/dense/bias/Adamsave_1/RestoreV2:31*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(
Ä
save_1/Assign_32Assignlayer_fc4/dense/bias/Adam_1save_1/RestoreV2:32*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:
Ć
save_1/Assign_33Assignlayer_fc4/dense/kernelsave_1/RestoreV2:33*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel
Ë
save_1/Assign_34Assignlayer_fc4/dense/kernel/Adamsave_1/RestoreV2:34*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel
Í
save_1/Assign_35Assignlayer_fc4/dense/kernel/Adam_1save_1/RestoreV2:35*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:

˝
save_1/Assign_36Assignlayer_fc5/dense/biassave_1/RestoreV2:36*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0
Â
save_1/Assign_37Assignlayer_fc5/dense/bias/Adamsave_1/RestoreV2:37*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:
Ä
save_1/Assign_38Assignlayer_fc5/dense/bias/Adam_1save_1/RestoreV2:38*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:
Ć
save_1/Assign_39Assignlayer_fc5/dense/kernelsave_1/RestoreV2:39*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel
Ë
save_1/Assign_40Assignlayer_fc5/dense/kernel/Adamsave_1/RestoreV2:40* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(
Í
save_1/Assign_41Assignlayer_fc5/dense/kernel/Adam_1save_1/RestoreV2:41*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Ź
save_1/restore_allNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9

init_1NoOp^beta1_power/Assign^beta2_power/Assign^dense/kernel/Adam/Assign^dense/kernel/Adam_1/Assign^dense/kernel/Assign^global_step/Assign!^layer_fc0/dense/bias/Adam/Assign#^layer_fc0/dense/bias/Adam_1/Assign^layer_fc0/dense/bias/Assign#^layer_fc0/dense/kernel/Adam/Assign%^layer_fc0/dense/kernel/Adam_1/Assign^layer_fc0/dense/kernel/Assign!^layer_fc1/dense/bias/Adam/Assign#^layer_fc1/dense/bias/Adam_1/Assign^layer_fc1/dense/bias/Assign#^layer_fc1/dense/kernel/Adam/Assign%^layer_fc1/dense/kernel/Adam_1/Assign^layer_fc1/dense/kernel/Assign!^layer_fc2/dense/bias/Adam/Assign#^layer_fc2/dense/bias/Adam_1/Assign^layer_fc2/dense/bias/Assign#^layer_fc2/dense/kernel/Adam/Assign%^layer_fc2/dense/kernel/Adam_1/Assign^layer_fc2/dense/kernel/Assign!^layer_fc3/dense/bias/Adam/Assign#^layer_fc3/dense/bias/Adam_1/Assign^layer_fc3/dense/bias/Assign#^layer_fc3/dense/kernel/Adam/Assign%^layer_fc3/dense/kernel/Adam_1/Assign^layer_fc3/dense/kernel/Assign!^layer_fc4/dense/bias/Adam/Assign#^layer_fc4/dense/bias/Adam_1/Assign^layer_fc4/dense/bias/Assign#^layer_fc4/dense/kernel/Adam/Assign%^layer_fc4/dense/kernel/Adam_1/Assign^layer_fc4/dense/kernel/Assign!^layer_fc5/dense/bias/Adam/Assign#^layer_fc5/dense/bias/Adam_1/Assign^layer_fc5/dense/bias/Assign#^layer_fc5/dense/kernel/Adam/Assign%^layer_fc5/dense/kernel/Adam_1/Assign^layer_fc5/dense/kernel/Assign
R
save_2/ConstConst*
_output_shapes
: *
valueB Bmodel*
dtype0

save_2/StringJoin/inputs_1Const*<
value3B1 B+_temp_b0df91efda6e442c82bb93ce3340a53d/part*
dtype0*
_output_shapes
: 
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 

save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
	
save_2/SaveV2/tensor_namesConst*ž
value´Bą*Bbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBlayer_fc0/dense/biasBlayer_fc0/dense/bias/AdamBlayer_fc0/dense/bias/Adam_1Blayer_fc0/dense/kernelBlayer_fc0/dense/kernel/AdamBlayer_fc0/dense/kernel/Adam_1Blayer_fc1/dense/biasBlayer_fc1/dense/bias/AdamBlayer_fc1/dense/bias/Adam_1Blayer_fc1/dense/kernelBlayer_fc1/dense/kernel/AdamBlayer_fc1/dense/kernel/Adam_1Blayer_fc2/dense/biasBlayer_fc2/dense/bias/AdamBlayer_fc2/dense/bias/Adam_1Blayer_fc2/dense/kernelBlayer_fc2/dense/kernel/AdamBlayer_fc2/dense/kernel/Adam_1Blayer_fc3/dense/biasBlayer_fc3/dense/bias/AdamBlayer_fc3/dense/bias/Adam_1Blayer_fc3/dense/kernelBlayer_fc3/dense/kernel/AdamBlayer_fc3/dense/kernel/Adam_1Blayer_fc4/dense/biasBlayer_fc4/dense/bias/AdamBlayer_fc4/dense/bias/Adam_1Blayer_fc4/dense/kernelBlayer_fc4/dense/kernel/AdamBlayer_fc4/dense/kernel/Adam_1Blayer_fc5/dense/biasBlayer_fc5/dense/bias/AdamBlayer_fc5/dense/bias/Adam_1Blayer_fc5/dense/kernelBlayer_fc5/dense/kernel/AdamBlayer_fc5/dense/kernel/Adam_1*
dtype0*
_output_shapes
:*
š
save_2/SaveV2/shape_and_slicesConst*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:*
Î	
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesbeta1_powerbeta2_powerdense/kerneldense/kernel/Adamdense/kernel/Adam_1global_steplayer_fc0/dense/biaslayer_fc0/dense/bias/Adamlayer_fc0/dense/bias/Adam_1layer_fc0/dense/kernellayer_fc0/dense/kernel/Adamlayer_fc0/dense/kernel/Adam_1layer_fc1/dense/biaslayer_fc1/dense/bias/Adamlayer_fc1/dense/bias/Adam_1layer_fc1/dense/kernellayer_fc1/dense/kernel/Adamlayer_fc1/dense/kernel/Adam_1layer_fc2/dense/biaslayer_fc2/dense/bias/Adamlayer_fc2/dense/bias/Adam_1layer_fc2/dense/kernellayer_fc2/dense/kernel/Adamlayer_fc2/dense/kernel/Adam_1layer_fc3/dense/biaslayer_fc3/dense/bias/Adamlayer_fc3/dense/bias/Adam_1layer_fc3/dense/kernellayer_fc3/dense/kernel/Adamlayer_fc3/dense/kernel/Adam_1layer_fc4/dense/biaslayer_fc4/dense/bias/Adamlayer_fc4/dense/bias/Adam_1layer_fc4/dense/kernellayer_fc4/dense/kernel/Adamlayer_fc4/dense/kernel/Adam_1layer_fc5/dense/biaslayer_fc5/dense/bias/Adamlayer_fc5/dense/bias/Adam_1layer_fc5/dense/kernellayer_fc5/dense/kernel/Adamlayer_fc5/dense/kernel/Adam_1*8
dtypes.
,2*

save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
Ł
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
_output_shapes
:*
T0*

axis 

save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(

save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
_output_shapes
: *
T0
	
save_2/RestoreV2/tensor_namesConst*ž
value´Bą*Bbeta1_powerBbeta2_powerBdense/kernelBdense/kernel/AdamBdense/kernel/Adam_1Bglobal_stepBlayer_fc0/dense/biasBlayer_fc0/dense/bias/AdamBlayer_fc0/dense/bias/Adam_1Blayer_fc0/dense/kernelBlayer_fc0/dense/kernel/AdamBlayer_fc0/dense/kernel/Adam_1Blayer_fc1/dense/biasBlayer_fc1/dense/bias/AdamBlayer_fc1/dense/bias/Adam_1Blayer_fc1/dense/kernelBlayer_fc1/dense/kernel/AdamBlayer_fc1/dense/kernel/Adam_1Blayer_fc2/dense/biasBlayer_fc2/dense/bias/AdamBlayer_fc2/dense/bias/Adam_1Blayer_fc2/dense/kernelBlayer_fc2/dense/kernel/AdamBlayer_fc2/dense/kernel/Adam_1Blayer_fc3/dense/biasBlayer_fc3/dense/bias/AdamBlayer_fc3/dense/bias/Adam_1Blayer_fc3/dense/kernelBlayer_fc3/dense/kernel/AdamBlayer_fc3/dense/kernel/Adam_1Blayer_fc4/dense/biasBlayer_fc4/dense/bias/AdamBlayer_fc4/dense/bias/Adam_1Blayer_fc4/dense/kernelBlayer_fc4/dense/kernel/AdamBlayer_fc4/dense/kernel/Adam_1Blayer_fc5/dense/biasBlayer_fc5/dense/bias/AdamBlayer_fc5/dense/bias/Adam_1Blayer_fc5/dense/kernelBlayer_fc5/dense/kernel/AdamBlayer_fc5/dense/kernel/Adam_1*
dtype0*
_output_shapes
:*
ź
!save_2/RestoreV2/shape_and_slicesConst*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:*
č
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*ž
_output_shapesŤ
¨::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*
Ą
save_2/AssignAssignbeta1_powersave_2/RestoreV2*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: 
Ľ
save_2/Assign_1Assignbeta2_powersave_2/RestoreV2:1*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
: 
Ż
save_2/Assign_2Assigndense/kernelsave_2/RestoreV2:2*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	
´
save_2/Assign_3Assigndense/kernel/Adamsave_2/RestoreV2:3*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	
ś
save_2/Assign_4Assigndense/kernel/Adam_1save_2/RestoreV2:4*
use_locking(*
T0*
_class
loc:@dense/kernel*
validate_shape(*
_output_shapes
:	
¤
save_2/Assign_5Assignglobal_stepsave_2/RestoreV2:5*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@global_step*
validate_shape(
ş
save_2/Assign_6Assignlayer_fc0/dense/biassave_2/RestoreV2:6*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias*
validate_shape(
ż
save_2/Assign_7Assignlayer_fc0/dense/bias/Adamsave_2/RestoreV2:7*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias
Á
save_2/Assign_8Assignlayer_fc0/dense/bias/Adam_1save_2/RestoreV2:8*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*'
_class
loc:@layer_fc0/dense/bias
Â
save_2/Assign_9Assignlayer_fc0/dense/kernelsave_2/RestoreV2:9*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 
É
save_2/Assign_10Assignlayer_fc0/dense/kernel/Adamsave_2/RestoreV2:10*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: *
use_locking(
Ë
save_2/Assign_11Assignlayer_fc0/dense/kernel/Adam_1save_2/RestoreV2:11*
use_locking(*
T0*)
_class
loc:@layer_fc0/dense/kernel*
validate_shape(*
_output_shapes

: 
ź
save_2/Assign_12Assignlayer_fc1/dense/biassave_2/RestoreV2:12*
T0*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
Á
save_2/Assign_13Assignlayer_fc1/dense/bias/Adamsave_2/RestoreV2:13*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@layer_fc1/dense/bias
Ă
save_2/Assign_14Assignlayer_fc1/dense/bias/Adam_1save_2/RestoreV2:14*'
_class
loc:@layer_fc1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
Ä
save_2/Assign_15Assignlayer_fc1/dense/kernelsave_2/RestoreV2:15*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @*
use_locking(
É
save_2/Assign_16Assignlayer_fc1/dense/kernel/Adamsave_2/RestoreV2:16*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @
Ë
save_2/Assign_17Assignlayer_fc1/dense/kernel/Adam_1save_2/RestoreV2:17*
use_locking(*
T0*)
_class
loc:@layer_fc1/dense/kernel*
validate_shape(*
_output_shapes

: @
˝
save_2/Assign_18Assignlayer_fc2/dense/biassave_2/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:
Â
save_2/Assign_19Assignlayer_fc2/dense/bias/Adamsave_2/RestoreV2:19*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ä
save_2/Assign_20Assignlayer_fc2/dense/bias/Adam_1save_2/RestoreV2:20*
use_locking(*
T0*'
_class
loc:@layer_fc2/dense/bias*
validate_shape(*
_output_shapes	
:
Ĺ
save_2/Assign_21Assignlayer_fc2/dense/kernelsave_2/RestoreV2:21*
_output_shapes
:	@*
use_locking(*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(
Ę
save_2/Assign_22Assignlayer_fc2/dense/kernel/Adamsave_2/RestoreV2:22*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@*
use_locking(
Ě
save_2/Assign_23Assignlayer_fc2/dense/kernel/Adam_1save_2/RestoreV2:23*
use_locking(*
T0*)
_class
loc:@layer_fc2/dense/kernel*
validate_shape(*
_output_shapes
:	@
˝
save_2/Assign_24Assignlayer_fc3/dense/biassave_2/RestoreV2:24*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(
Â
save_2/Assign_25Assignlayer_fc3/dense/bias/Adamsave_2/RestoreV2:25*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(
Ä
save_2/Assign_26Assignlayer_fc3/dense/bias/Adam_1save_2/RestoreV2:26*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc3/dense/bias*
validate_shape(
Ć
save_2/Assign_27Assignlayer_fc3/dense/kernelsave_2/RestoreV2:27*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:

Ë
save_2/Assign_28Assignlayer_fc3/dense/kernel/Adamsave_2/RestoreV2:28*
use_locking(*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:

Í
save_2/Assign_29Assignlayer_fc3/dense/kernel/Adam_1save_2/RestoreV2:29*
T0*)
_class
loc:@layer_fc3/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(
˝
save_2/Assign_30Assignlayer_fc4/dense/biassave_2/RestoreV2:30*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Â
save_2/Assign_31Assignlayer_fc4/dense/bias/Adamsave_2/RestoreV2:31*
T0*'
_class
loc:@layer_fc4/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ä
save_2/Assign_32Assignlayer_fc4/dense/bias/Adam_1save_2/RestoreV2:32*
validate_shape(*
_output_shapes	
:*
use_locking(*
T0*'
_class
loc:@layer_fc4/dense/bias
Ć
save_2/Assign_33Assignlayer_fc4/dense/kernelsave_2/RestoreV2:33* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(
Ë
save_2/Assign_34Assignlayer_fc4/dense/kernel/Adamsave_2/RestoreV2:34*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
Í
save_2/Assign_35Assignlayer_fc4/dense/kernel/Adam_1save_2/RestoreV2:35*)
_class
loc:@layer_fc4/dense/kernel*
validate_shape(* 
_output_shapes
:
*
use_locking(*
T0
˝
save_2/Assign_36Assignlayer_fc5/dense/biassave_2/RestoreV2:36*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Â
save_2/Assign_37Assignlayer_fc5/dense/bias/Adamsave_2/RestoreV2:37*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:*
use_locking(
Ä
save_2/Assign_38Assignlayer_fc5/dense/bias/Adam_1save_2/RestoreV2:38*
use_locking(*
T0*'
_class
loc:@layer_fc5/dense/bias*
validate_shape(*
_output_shapes	
:
Ć
save_2/Assign_39Assignlayer_fc5/dense/kernelsave_2/RestoreV2:39*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:

Ë
save_2/Assign_40Assignlayer_fc5/dense/kernel/Adamsave_2/RestoreV2:40*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(* 
_output_shapes
:

Í
save_2/Assign_41Assignlayer_fc5/dense/kernel/Adam_1save_2/RestoreV2:41* 
_output_shapes
:
*
use_locking(*
T0*)
_class
loc:@layer_fc5/dense/kernel*
validate_shape(
Ž
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_5^save_2/Assign_6^save_2/Assign_7^save_2/Assign_8^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard "B
save_2/Const:0save_2/Identity:0save_2/restore_all (5 @F8"
train_op

Adam"ň
cond_contextňň
ç
dropout/cond/cond_textdropout/cond/pred_id:0dropout/cond/switch_t:0 *
dropout/cond/dropout/Floor:0
#dropout/cond/dropout/Shape/Switch:1
dropout/cond/dropout/Shape:0
dropout/cond/dropout/add:0
dropout/cond/dropout/div:0
 dropout/cond/dropout/keep_prob:0
dropout/cond/dropout/mul:0
3dropout/cond/dropout/random_uniform/RandomUniform:0
)dropout/cond/dropout/random_uniform/max:0
)dropout/cond/dropout/random_uniform/min:0
)dropout/cond/dropout/random_uniform/mul:0
)dropout/cond/dropout/random_uniform/sub:0
%dropout/cond/dropout/random_uniform:0
dropout/cond/pred_id:0
dropout/cond/switch_t:0
layer_fc5/dense/Relu:00
dropout/cond/pred_id:0dropout/cond/pred_id:0=
layer_fc5/dense/Relu:0#dropout/cond/dropout/Shape/Switch:1
ź
dropout/cond/cond_text_1dropout/cond/pred_id:0dropout/cond/switch_f:0*î
dropout/cond/Identity/Switch:0
dropout/cond/Identity:0
dropout/cond/pred_id:0
dropout/cond/switch_f:0
layer_fc5/dense/Relu:08
layer_fc5/dense/Relu:0dropout/cond/Identity/Switch:00
dropout/cond/pred_id:0dropout/cond/pred_id:0

@mean_squared_error/assert_broadcastable/is_valid_shape/cond_text@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0 *Á
3mean_squared_error/assert_broadcastable/is_scalar:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:0
Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
Amean_squared_error/assert_broadcastable/is_valid_shape/switch_t:0x
3mean_squared_error/assert_broadcastable/is_scalar:0Amean_squared_error/assert_broadcastable/is_valid_shape/Switch_1:1
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
éI
Bmean_squared_error/assert_broadcastable/is_valid_shape/cond_text_1@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0Amean_squared_error/assert_broadcastable/is_valid_shape/switch_f:0*ľ!
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
Xmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
Ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
omean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
jmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
emean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0
Amean_squared_error/assert_broadcastable/is_valid_shape/switch_f:0
5mean_squared_error/assert_broadcastable/values/rank:0
6mean_squared_error/assert_broadcastable/values/shape:0
6mean_squared_error/assert_broadcastable/weights/rank:0
7mean_squared_error/assert_broadcastable/weights/shape:0
@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0@mean_squared_error/assert_broadcastable/is_valid_shape/pred_id:0˛
7mean_squared_error/assert_broadcastable/weights/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
5mean_squared_error/assert_broadcastable/values/rank:0fmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0Ż
6mean_squared_error/assert_broadcastable/values/shape:0umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0˘
6mean_squared_error/assert_broadcastable/weights/rank:0hmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:02ő
ň
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textZmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *Ř
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
|mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
rmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
nmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
pmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
omean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
jmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
tmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
smean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
mmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
emean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
cmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
6mean_squared_error/assert_broadcastable/values/shape:0
7mean_squared_error/assert_broadcastable/weights/shape:0´
7mean_squared_error/assert_broadcastable/weights/shape:0ymean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1ň
wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0ą
6mean_squared_error/assert_broadcastable/values/shape:0wmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1î
umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0umean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0¸
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:02í
ę
\mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*Đ
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0¸
Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0Zmean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ž
_mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0[mean_squared_error/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0

=mean_squared_error/assert_broadcastable/AssertGuard/cond_text=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0 *É
Hmean_squared_error/assert_broadcastable/AssertGuard/control_dependency:0
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
>mean_squared_error/assert_broadcastable/AssertGuard/switch_t:0~
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
ś
?mean_squared_error/assert_broadcastable/AssertGuard/cond_text_1=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0>mean_squared_error/assert_broadcastable/AssertGuard/switch_f:0*ó
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2:0
Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_0:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_1:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_2:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_4:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_5:0
Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/data_7:0
Jmean_squared_error/assert_broadcastable/AssertGuard/control_dependency_1:0
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
>mean_squared_error/assert_broadcastable/AssertGuard/switch_f:0
3mean_squared_error/assert_broadcastable/is_scalar:0
>mean_squared_error/assert_broadcastable/is_valid_shape/Merge:0
6mean_squared_error/assert_broadcastable/values/shape:0
7mean_squared_error/assert_broadcastable/weights/shape:0
7mean_squared_error/assert_broadcastable/weights/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_1:0~
=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0=mean_squared_error/assert_broadcastable/AssertGuard/pred_id:0
6mean_squared_error/assert_broadcastable/values/shape:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_2:0|
3mean_squared_error/assert_broadcastable/is_scalar:0Emean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch_3:0
>mean_squared_error/assert_broadcastable/is_valid_shape/Merge:0Cmean_squared_error/assert_broadcastable/AssertGuard/Assert/Switch:0
ö
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0 *Đ
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_t:0´
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Switch_1:1Ŕ
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
Ů`
`mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/cond_text_1^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0*,
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:0
vmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Merge:1
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:0
wmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch:1
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:0
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0
_mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/switch_f:0
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0ď
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0Ü
Smean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/rank:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch:0Ŕ
^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0^mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/pred_id:0ě
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0ß
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/rank:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank/Switch_1:02Ć&
Ă&
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_textxmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0 *Ď#
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:1
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/DenseToDenseSetOperation:2
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/dim:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/dim:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat/axis:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/concat:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/num_invalid_dims:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Const:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like/Shape:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ones_like:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/x:0
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_t:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0Ź
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch:0î
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims/Switch_1:1ń
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch_1:1°
mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/has_invalid_dims/ExpandDims_1/Switch:0ô
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:02Ő
Ň
zmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/cond_text_1xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0*Ţ
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:1
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/switch_f:0ú
}mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/is_same_rank:0ymean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/Switch_1:0ô
xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0xmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/has_valid_nonscalar_shape/pred_id:0
ý
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0 *ŕ
fmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency:0
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_t:0ş
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
˝
]mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/cond_text_1[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0* 
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0
cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_0:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_1:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_2:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_4:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_5:0
amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/data_7:0
hmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/control_dependency_1:0
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/switch_f:0
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0Á
\mean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_valid_shape/Merge:0amean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch:0ź
Umean_squared_error/num_present/broadcast_weights/assert_broadcastable/weights/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_1:0ş
[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0[mean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/pred_id:0¸
Qmean_squared_error/num_present/broadcast_weights/assert_broadcastable/is_scalar:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_3:0ť
Tmean_squared_error/num_present/broadcast_weights/assert_broadcastable/values/shape:0cmean_squared_error/num_present/broadcast_weights/assert_broadcastable/AssertGuard/Assert/Switch_2:0"Ű
regularization_lossesÁ
ž
3layer_fc0/dense/kernel/Regularizer/l2_regularizer:0
3layer_fc1/dense/kernel/Regularizer/l2_regularizer:0
3layer_fc2/dense/kernel/Regularizer/l2_regularizer:0
3layer_fc3/dense/kernel/Regularizer/l2_regularizer:0
3layer_fc4/dense/kernel/Regularizer/l2_regularizer:0
3layer_fc5/dense/kernel/Regularizer/l2_regularizer:0":
	summaries-
+
loss:0
learning_rate:0
histogram_loss:0"đ
trainable_variablesŘŐ

layer_fc0/dense/kernel:0layer_fc0/dense/kernel/Assignlayer_fc0/dense/kernel/read:023layer_fc0/dense/kernel/Initializer/random_uniform:08
~
layer_fc0/dense/bias:0layer_fc0/dense/bias/Assignlayer_fc0/dense/bias/read:02(layer_fc0/dense/bias/Initializer/zeros:08

layer_fc1/dense/kernel:0layer_fc1/dense/kernel/Assignlayer_fc1/dense/kernel/read:023layer_fc1/dense/kernel/Initializer/random_uniform:08
~
layer_fc1/dense/bias:0layer_fc1/dense/bias/Assignlayer_fc1/dense/bias/read:02(layer_fc1/dense/bias/Initializer/zeros:08

layer_fc2/dense/kernel:0layer_fc2/dense/kernel/Assignlayer_fc2/dense/kernel/read:023layer_fc2/dense/kernel/Initializer/random_uniform:08
~
layer_fc2/dense/bias:0layer_fc2/dense/bias/Assignlayer_fc2/dense/bias/read:02(layer_fc2/dense/bias/Initializer/zeros:08

layer_fc3/dense/kernel:0layer_fc3/dense/kernel/Assignlayer_fc3/dense/kernel/read:023layer_fc3/dense/kernel/Initializer/random_uniform:08
~
layer_fc3/dense/bias:0layer_fc3/dense/bias/Assignlayer_fc3/dense/bias/read:02(layer_fc3/dense/bias/Initializer/zeros:08

layer_fc4/dense/kernel:0layer_fc4/dense/kernel/Assignlayer_fc4/dense/kernel/read:023layer_fc4/dense/kernel/Initializer/random_uniform:08
~
layer_fc4/dense/bias:0layer_fc4/dense/bias/Assignlayer_fc4/dense/bias/read:02(layer_fc4/dense/bias/Initializer/zeros:08

layer_fc5/dense/kernel:0layer_fc5/dense/kernel/Assignlayer_fc5/dense/kernel/read:023layer_fc5/dense/kernel/Initializer/random_uniform:08
~
layer_fc5/dense/bias:0layer_fc5/dense/bias/Assignlayer_fc5/dense/bias/read:02(layer_fc5/dense/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08"Č.
	variablesş.ˇ.
@
global_step:0global_step/Assignglobal_step/read:02Const:0

layer_fc0/dense/kernel:0layer_fc0/dense/kernel/Assignlayer_fc0/dense/kernel/read:023layer_fc0/dense/kernel/Initializer/random_uniform:08
~
layer_fc0/dense/bias:0layer_fc0/dense/bias/Assignlayer_fc0/dense/bias/read:02(layer_fc0/dense/bias/Initializer/zeros:08

layer_fc1/dense/kernel:0layer_fc1/dense/kernel/Assignlayer_fc1/dense/kernel/read:023layer_fc1/dense/kernel/Initializer/random_uniform:08
~
layer_fc1/dense/bias:0layer_fc1/dense/bias/Assignlayer_fc1/dense/bias/read:02(layer_fc1/dense/bias/Initializer/zeros:08

layer_fc2/dense/kernel:0layer_fc2/dense/kernel/Assignlayer_fc2/dense/kernel/read:023layer_fc2/dense/kernel/Initializer/random_uniform:08
~
layer_fc2/dense/bias:0layer_fc2/dense/bias/Assignlayer_fc2/dense/bias/read:02(layer_fc2/dense/bias/Initializer/zeros:08

layer_fc3/dense/kernel:0layer_fc3/dense/kernel/Assignlayer_fc3/dense/kernel/read:023layer_fc3/dense/kernel/Initializer/random_uniform:08
~
layer_fc3/dense/bias:0layer_fc3/dense/bias/Assignlayer_fc3/dense/bias/read:02(layer_fc3/dense/bias/Initializer/zeros:08

layer_fc4/dense/kernel:0layer_fc4/dense/kernel/Assignlayer_fc4/dense/kernel/read:023layer_fc4/dense/kernel/Initializer/random_uniform:08
~
layer_fc4/dense/bias:0layer_fc4/dense/bias/Assignlayer_fc4/dense/bias/read:02(layer_fc4/dense/bias/Initializer/zeros:08

layer_fc5/dense/kernel:0layer_fc5/dense/kernel/Assignlayer_fc5/dense/kernel/read:023layer_fc5/dense/kernel/Initializer/random_uniform:08
~
layer_fc5/dense/bias:0layer_fc5/dense/bias/Assignlayer_fc5/dense/bias/read:02(layer_fc5/dense/bias/Initializer/zeros:08
g
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0

layer_fc0/dense/kernel/Adam:0"layer_fc0/dense/kernel/Adam/Assign"layer_fc0/dense/kernel/Adam/read:02/layer_fc0/dense/kernel/Adam/Initializer/zeros:0
 
layer_fc0/dense/kernel/Adam_1:0$layer_fc0/dense/kernel/Adam_1/Assign$layer_fc0/dense/kernel/Adam_1/read:021layer_fc0/dense/kernel/Adam_1/Initializer/zeros:0

layer_fc0/dense/bias/Adam:0 layer_fc0/dense/bias/Adam/Assign layer_fc0/dense/bias/Adam/read:02-layer_fc0/dense/bias/Adam/Initializer/zeros:0

layer_fc0/dense/bias/Adam_1:0"layer_fc0/dense/bias/Adam_1/Assign"layer_fc0/dense/bias/Adam_1/read:02/layer_fc0/dense/bias/Adam_1/Initializer/zeros:0

layer_fc1/dense/kernel/Adam:0"layer_fc1/dense/kernel/Adam/Assign"layer_fc1/dense/kernel/Adam/read:02/layer_fc1/dense/kernel/Adam/Initializer/zeros:0
 
layer_fc1/dense/kernel/Adam_1:0$layer_fc1/dense/kernel/Adam_1/Assign$layer_fc1/dense/kernel/Adam_1/read:021layer_fc1/dense/kernel/Adam_1/Initializer/zeros:0

layer_fc1/dense/bias/Adam:0 layer_fc1/dense/bias/Adam/Assign layer_fc1/dense/bias/Adam/read:02-layer_fc1/dense/bias/Adam/Initializer/zeros:0

layer_fc1/dense/bias/Adam_1:0"layer_fc1/dense/bias/Adam_1/Assign"layer_fc1/dense/bias/Adam_1/read:02/layer_fc1/dense/bias/Adam_1/Initializer/zeros:0

layer_fc2/dense/kernel/Adam:0"layer_fc2/dense/kernel/Adam/Assign"layer_fc2/dense/kernel/Adam/read:02/layer_fc2/dense/kernel/Adam/Initializer/zeros:0
 
layer_fc2/dense/kernel/Adam_1:0$layer_fc2/dense/kernel/Adam_1/Assign$layer_fc2/dense/kernel/Adam_1/read:021layer_fc2/dense/kernel/Adam_1/Initializer/zeros:0

layer_fc2/dense/bias/Adam:0 layer_fc2/dense/bias/Adam/Assign layer_fc2/dense/bias/Adam/read:02-layer_fc2/dense/bias/Adam/Initializer/zeros:0

layer_fc2/dense/bias/Adam_1:0"layer_fc2/dense/bias/Adam_1/Assign"layer_fc2/dense/bias/Adam_1/read:02/layer_fc2/dense/bias/Adam_1/Initializer/zeros:0

layer_fc3/dense/kernel/Adam:0"layer_fc3/dense/kernel/Adam/Assign"layer_fc3/dense/kernel/Adam/read:02/layer_fc3/dense/kernel/Adam/Initializer/zeros:0
 
layer_fc3/dense/kernel/Adam_1:0$layer_fc3/dense/kernel/Adam_1/Assign$layer_fc3/dense/kernel/Adam_1/read:021layer_fc3/dense/kernel/Adam_1/Initializer/zeros:0

layer_fc3/dense/bias/Adam:0 layer_fc3/dense/bias/Adam/Assign layer_fc3/dense/bias/Adam/read:02-layer_fc3/dense/bias/Adam/Initializer/zeros:0

layer_fc3/dense/bias/Adam_1:0"layer_fc3/dense/bias/Adam_1/Assign"layer_fc3/dense/bias/Adam_1/read:02/layer_fc3/dense/bias/Adam_1/Initializer/zeros:0

layer_fc4/dense/kernel/Adam:0"layer_fc4/dense/kernel/Adam/Assign"layer_fc4/dense/kernel/Adam/read:02/layer_fc4/dense/kernel/Adam/Initializer/zeros:0
 
layer_fc4/dense/kernel/Adam_1:0$layer_fc4/dense/kernel/Adam_1/Assign$layer_fc4/dense/kernel/Adam_1/read:021layer_fc4/dense/kernel/Adam_1/Initializer/zeros:0

layer_fc4/dense/bias/Adam:0 layer_fc4/dense/bias/Adam/Assign layer_fc4/dense/bias/Adam/read:02-layer_fc4/dense/bias/Adam/Initializer/zeros:0

layer_fc4/dense/bias/Adam_1:0"layer_fc4/dense/bias/Adam_1/Assign"layer_fc4/dense/bias/Adam_1/read:02/layer_fc4/dense/bias/Adam_1/Initializer/zeros:0

layer_fc5/dense/kernel/Adam:0"layer_fc5/dense/kernel/Adam/Assign"layer_fc5/dense/kernel/Adam/read:02/layer_fc5/dense/kernel/Adam/Initializer/zeros:0
 
layer_fc5/dense/kernel/Adam_1:0$layer_fc5/dense/kernel/Adam_1/Assign$layer_fc5/dense/kernel/Adam_1/read:021layer_fc5/dense/kernel/Adam_1/Initializer/zeros:0

layer_fc5/dense/bias/Adam:0 layer_fc5/dense/bias/Adam/Assign layer_fc5/dense/bias/Adam/read:02-layer_fc5/dense/bias/Adam/Initializer/zeros:0

layer_fc5/dense/bias/Adam_1:0"layer_fc5/dense/bias/Adam_1/Assign"layer_fc5/dense/bias/Adam_1/read:02/layer_fc5/dense/bias/Adam_1/Initializer/zeros:0
p
dense/kernel/Adam:0dense/kernel/Adam/Assigndense/kernel/Adam/read:02%dense/kernel/Adam/Initializer/zeros:0
x
dense/kernel/Adam_1:0dense/kernel/Adam_1/Assigndense/kernel/Adam_1/read:02'dense/kernel/Adam_1/Initializer/zeros:0"(
losses

mean_squared_error/value:0*Đ
serving_defaultź
.
model_istraining
model_istraining:0

3
model_input$
model_input:0˙˙˙˙˙˙˙˙˙9
model_prediction%
model_prediction:0˙˙˙˙˙˙˙˙˙tensorflow/serving/predict