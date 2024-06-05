// ensure all content is loaded before running the script
document.addEventListener('DOMContentLoaded', (_event) => {

	async function doTraining(model) {
		const totalEpochs = 10;
		const progressBar = document.getElementById('progress-bar');
		const epochDisplay = document.getElementById('epoch');
		const lossDisplay = document.getElementById('loss');

		const history = await model.fit(xs, ys, {
			epochs: totalEpochs,
			callbacks: {
				onEpochEnd: async (epoch, logs) => {
					console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss);
					epochDisplay.innerText = (epoch + 1).toString();
					lossDisplay.innerText = logs.loss.toFixed(4).toString();
					const progress = ((epoch + 1) / totalEpochs) * 100;
					progressBar.style.width = progress + '%';
				}
			}
		});

		return history;
	}

	const model = tf.sequential();
	model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

	model.compile({
		loss: 'meanSquaredError',
		optimizer: 'sgd'
	});

	const xs = tf.tensor2d([1, 2, 3, 4], [4, 1]);
	const ys = tf.tensor2d([1, 3, 5, 7], [4, 1]);

	doTraining(model).then(history => {
		console.log('Training complete');
		console.log('Loss:', history.history.loss[0]);
	});
	
});
