from keras.applications import mobilenet

model = mobilenet.MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, include_top=True, weights='imagenet', input_tensor=None, pooling=None, classes=1000)
model.save_weights('mobile.hdf5')
with open('mobile.json','w') as f:
	f.write(model.to_json())