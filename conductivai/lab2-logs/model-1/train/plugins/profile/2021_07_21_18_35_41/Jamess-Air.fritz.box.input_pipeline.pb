	ffffff??ffffff??!ffffff??	6??9 @6??9 @!6??9 @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ffffff???(\?????AB`??"???Y;?O??n??*	     `b@2F
Iterator::Model333333??!?????I@)?~j?t???1????S@@:Preprocessing2d
-Iterator::Model::ParallelMap::Zip[0]::FlatMapˡE?????!?kv?"?;@)ˡE?????1?kv?"?;@:Preprocessing2S
Iterator::Model::ParallelMap???S㥛?!?S?r
^2@)???S㥛?1?S?r
^2@:Preprocessing2j
3Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat???S㥛?!?S?r
^2@)????????1@??ҽ1@:Preprocessing2?
MIterator::Model::ParallelMap::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice????Mbp?!q?{??@)????Mbp?1q?{??@:Preprocessing2v
?Iterator::Model::ParallelMap::Zip[1]::ForeverRepeat::FromTensor????Mb`?!q?{????)????Mb`?1q?{????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 8.1% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2B22.0 % of the total step time sampled is spent on All Others time.>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?(\??????(\?????!?(\?????      ??!       "      ??!       *      ??!       2	B`??"???B`??"???!B`??"???:      ??!       B      ??!       J	;?O??n??;?O??n??!;?O??n??R      ??!       Z	;?O??n??;?O??n??!;?O??n??JCPU_ONLY