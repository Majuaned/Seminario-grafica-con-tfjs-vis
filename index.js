const model = tf.sequential();

async function entrenar() {
    let epocas = parseInt(document.getElementById("txtEpocas").value);
    const epochs = epocas;
    const datos = []


    // Create a simple model. Esto me va a crear lo que se vió el otro día, capas neuronales que procesan y envian los resultados a otras neuronas o no...

    model.add(tf.layers.dense({units: 1, inputShape: [1]}));
    console.log(epocas)
    
    // Prepare the model for training: Specify the loss and the optimizer...
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});
    
    // Generate some synthetic data for training. (y = 2x - 1)
    const xs = tf.tensor2d([-1, 0, 1, 2, 3, 4], [6, 1]);
    const ys = tf.tensor2d([-3, -1, 1, 3, 5, 7], [6, 1]);
    
    // Train the model using the data.
    //****************************************** */

    //await model.fit(xs, ys, {epochs: x});

    const history = await model.fit(xs, ys, {
        epochs: epochs,
        callbacks: {
          onEpochEnd: (epoch, logs) => {
             console.log(logs);
            
             console.log(`Epoch ${epoch+1} - Loss: ${logs.loss.toFixed(4)}`);

              datos.push({index:epoch+1,value:logs.loss.toFixed(4)});
            
            // Get a surface
            const surface = tfvis.visor().surface({ name: 'Barchart', tab: 'Charts' });
            
            // Render a barchart on that surface
            tfvis.render.barchart(surface, datos, {});
          }
        }
      });

      // Imprimir la pérdida final
      console.log(`Final Loss: ${history.history.loss[epochs-1].toFixed(4)}`);
     
      alert("termino de entrenar");

    //******************************************* */
    
    //console.log(repeticiones)
    
    //console.log(model.predict(tf.tensor2d([20],[1,1])).dataSync())
    
    // document.getElementById('micro-out-div').innerText =
    // model.predict(tf.tensor2d([20], [1, 1])).dataSync();
    // 


}

function predecir() {
    let x2 = parseInt(document.getElementById("txtNroPredecir").value);

    document.getElementById('micro-out-div').innerText =
    model.predict(tf.tensor2d([x2], [1, 1])).dataSync();
};