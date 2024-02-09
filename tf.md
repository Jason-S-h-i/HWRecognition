# TensorFlow API分析与应用

## tf.keras

Keras API 的实现，tensorflow的高层API

### 模型的API

#### The Model class
一个模型将所有的层聚集成一个有训练和推断特征的对象

参数：
- 输入：一个keras.Input对象
- 输出：一个张量
- 名字：字符串，模型的名字

有两种方法可以实例化一个模型：

1. 使用函数形式的API
2. 创建一个子类

summary方法

```python
Model.summary(
    line_length=None,
    positions=None,
    print_fn=None,
    expand_nested=False,
    show_trainable=False,
    layer_range=None,
)
```

get_layer方法

```python
Model.get_layer(name=None, index=None)
```



#### The Sequential class
Sequential将一个多层先行网络聚合进tf.keras.Model

add方法

`Sequential.add(layer)`

在层堆的最上方加入一个层

参数
- layer：层实例

pop方法

`Sequential.pop()`

移走模型的最后一层

#### Model training APIs

compile方法

确认训练的模型参数

```python
Model.compile(
    optimizer="rmsprop",
    loss=None,
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None,
    pss_evaluation_shards=0,
    **kwargs
)
```

fit方法

```python
Model.fit(
    x=None,
    y=None,
    batch_size=None,
    epochs=1,
    verbose="auto",
    callbacks=None,
    validation_split=0.0,
    validation_data=None,
    shuffle=True,
    class_weight=None,
    sample_weight=None,
    initial_epoch=0,
    steps_per_epoch=None,
    validation_steps=None,
    validation_batch_size=None,
    validation_freq=1,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
```
训练模型进行固定次数的epoch(数据集迭代)。

evaluate方法

```python
Model.evaluate(
    x=None,
    y=None,
    batch_size=None,
    verbose="auto",
    sample_weight=None,
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
    return_dict=False,
    **kwargs
)
```

predict方法

```python
Model.predict(
    x,
    batch_size=None,
    verbose="auto",
    steps=None,
    callbacks=None,
    max_queue_size=10,
    workers=1,
    use_multiprocessing=False,
)
```

train_on_batch方法

```python
Model.train_on_batch(
    x,
    y=None,
    sample_weight=None,
    class_weight=None,
    reset_metrics=True,
    return_dict=False,
)
```

test_on_batch方法

```python
Model.test_on_batch(
    x, y=None, sample_weight=None, reset_metrics=True, return_dict=False
)
```

predict_on_batch方法

```python
Model.predict_on_batch(x)
```

run_eagerly方法

```python
tf.keras.Model.run_eagerly
```

### 层的API

#### The base Layer class 基础层类

Layer class

```python
tf.keras.layers.Layer(
    trainable=True, name=None, dtype=None, dynamic=False, **kwargs
)
```
所用的层都继承此种类

#### Layer activations 层激活函数

#### Layer weight initializers 层权重初始化方法

#### Layer weight regularizers 层权重正则化器

#### Layer weight constraints 层权重限制

#### Core layers 核心层

#### Convolution layers 卷积层

#### Pooling layers 池化层



## 网络的实现方式

```python
# Build the model
mnist_model = tf.keras.Sequential([
  tf.keras.layers.Conv2D(16,[3,3], activation='relu',
                         input_shape=(None, None, 1)),
  tf.keras.layers.Conv2D(16,[3,3], activation='relu'),
  tf.keras.layers.GlobalAveragePooling2D(),
  tf.keras.layers.Dense(10)
])
```