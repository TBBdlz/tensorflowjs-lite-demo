document.addEventListener('DOMContentLoaded', (event) => {
	let model;
	const canvas = document.getElementById('canvas');
	const ctx = canvas.getContext('2d');
	let mousePressed = false;

	canvas.addEventListener('mousedown', () => { mousePressed = true; });
	canvas.addEventListener('mouseup', () => { mousePressed = false; });
	canvas.addEventListener('mousemove', draw);

	async function loadModel() {
		try {
			model = await tflite.loadTFLiteModel('model.tflite');
			console.log('Model loaded');
		} catch (error) {
			console.error('Error loading the model:', error);
		}
	}

	function draw(event) {
		if (!mousePressed) return;
		const rect = canvas.getBoundingClientRect();
		const x = event.clientX - rect.left;
		const y = event.clientY - rect.top;
		ctx.fillStyle = 'black';
		ctx.fillRect(x, y, 10, 10);
	}

	function clearCanvas() {
		ctx.clearRect(0, 0, canvas.width, canvas.height);
	}

	async function classifyDrawing() {
		if (!model) {
			console.error('Model not loaded yet');
			return;
		}
		const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
		const input = tf.browser.fromPixels(imageData, 1).resizeBilinear([28, 28]).expandDims(0).div(255.0);
		try {
			const output = await model.predict(input);
			const predictions = Array.from(output.dataSync());
			const predictedLabel = predictions.indexOf(Math.max(...predictions));
			document.getElementById('result').innerText = `Predicted Label: ${predictedLabel}`;
		} catch (error) {
			console.error('Error during prediction:', error);
		}
	}

	// Load the model when the page loads
	loadModel();

	// Expose functions to global scope
	window.clearCanvas = clearCanvas;
	window.classifyDrawing = classifyDrawing;
});
