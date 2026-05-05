# Logging Recommendation based on Deep Learning and Block Dependency Graph


Logging is widely used in industrial practice to record software runtime behaviors, assisting developers in diagnosing issues such as error tracing and anomaly detection. Existing approaches focus on recommending logging in a function without considering inter-function characteristics. To this end, we first conduct an empirical study on 99 projects to identify key features that may impact developers’ decisions on logging. Next, we propose GraphLog, a novel approach that effectively fuses these key features for logging recommendations. Specifically, we construct a block dependency graph from invocation relationships between code blocks to extract semantic, syntactic, thematic, and structural features. As a result, we built a dataset with 91,700 samples. These samples are fed into a deep learning model that combines graph convolutional networks (GCNs) and long short-term memory (LSTM) networks. GraphLog can recommend logging by learning logging specifications across both spatial and temporal domains. To evaluate its effectiveness, we evaluate GraphLog on seven real-world projects. Experimental results demonstrate that GraphLog obtains 77.23% F1 when recommending logging, which is 2.64% higher than existing approaches.

## GraphLog model and dataset

### GraphLog Model

We present the deep learning model of GraphLog in the following code.

The model：

```python
1  # multiple feature learning model                                                                      
2  def get_graph_model_4(graph_leng,syntactic_len,method_name_x, method_name_y,                        
3                        num_word,embedding_dim):                                                      
4  #     model1 = tf.keras.models.Sequential([                                                         
5  #         Input(shape=(semantic_vec,1)),                                                            
6  #         (Flatten()),                                                                              
7  #         (keras.layers.Embedding(num_word, embedding_dim)),                                        
8  #         (keras.layers.GlobalMaxPool1D()),                                                         
9  #     ])                                                                                            
10     model2 = tf.keras.models.Sequential([                                                           
11         Input(shape=(graph_leng,1)),                                                                
12         Flatten(),                                                                                  
13     ])                                                                                              
14     model3 = tf.keras.models.Sequential([                                                           
15         Input(shape=(syntactic_len,1)),                                                             
16         (Flatten()),                                                                                
17         (keras.layers.Embedding(num_word, embedding_dim)),                                          
18         (keras.layers.GlobalMaxPool1D()),                                                           
19     ])                                                                                              
20     model4 = tf.keras.models.Sequential([                                                           
21         keras.layers.Masking(mask_value=0, input_shape=(method_name_x, method_name_y)),             
22         LSTM(units=8, activation='tanh', return_sequences=True),                                    
23         keras.layers.GlobalMaxPool1D(),                                                             
24                                                                                                     
25     ])                                                                                              
26     model_together = keras.layers.concatenate([ model2.output, model3.output,model4.output])        
27     model_together=Lambda(lambda x:keras.backend.expand_dims(x,axis=1))(model_together)             
28     x = LSTM(units=32, activation='tanh', return_sequences=True)(model_together)                    
29     x=keras.layers.GlobalMaxPool1D()(x)                                                             
30     final_output = Dropout(0.2)(x)                                                                  
31     final_output = Dense(2, activation='softmax')(final_output)                                     
32     model_together = Model(inputs=[model2.input, model3.input,model4.input], outputs=final_output)  
33     return model_together                                                                           
34 # graph Embedding layers
35 kernel_initializer="glorot_uniform"                                           
36 bias = True                                                                   
37 bias_initializer="zeros"                                                      
38 n_features = features_input.shape[2]                                          
39 n_nodes = features_input.shape[1]                                             
40 # Initialise input layers                                                     
41 x_features = Input(batch_shape=(1, n_nodes, n_features))                      
42 print(x_features.shape)                                                       
43 x_indices = Input(batch_shape=(1, None), dtype="int32")                       
44 x_adjacency = Input(batch_shape=(1, n_nodes, n_nodes))                        
45 x_inp = [x_features, x_indices, x_adjacency]                                  
46                                                                               
47 x = Dropout(0.5)(x_features)                                                  
48 x = GraphConvolution(32, activation='relu',                                   
49                      use_bias=True,                                           
50                      kernel_initializer=kernel_initializer,                   
51                      bias_initializer=bias_initializer)([x, x_adjacency])     
52 x = Dropout(0.5)(x)                                                           
53 x = GraphConvolution(32, activation='relu',                                   
54                      use_bias=True,                                           
55                      kernel_initializer=kernel_initializer,                   
56                      bias_initializer=bias_initializer)([x, x_adjacency])     
57 # x=LSTM(units=32, activation='tanh', return_sequences=True)(x)               
58 # x=(keras.layers.GlobalMaxPool1D())(x)                                       
59 # x=Flatten()(x)                                                              
60 x = GatherIndices(batch_dims=1)([x, x_indices])                               
61                                                                               
62                                                                               
63 output = Dense(1, activation='sigmoid')(x)                                    
64                                                                               
65 model = Model(inputs=[x_features, x_indices, x_adjacency], outputs=output)    
```


Compile model
```
1  model1.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),loss=losses.binary_crossentropy,metrics=["accuracy"],)
2  model2.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
```

GraphLog Dataset and models

This is the source code of papers: Automating Logging Location Recommendation in Software Engineering: A Deep Learning Model with Block Dependency Graphs and Heterogeneous Feature Fusion.The directory of manual study is the data of the empirical study results. The other directory is the source code which is used in the experiments.
