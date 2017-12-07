

const model = new KerasJS.Model({
	filepaths:{
		model: './mobile.json',
		weights: './mobile_weights.buf',
		metadata: './mobile_metadata.json'
	},
	gpu: false
})

model.read().then(()=>{
	const inputData = {
		'input_1': new Float32Array(data)
	}

	return model.predict(inputData)
})