import * as knnClassifier from "@tensorflow-models/knn-classifier";
import * as mobilenet from "@tensorflow-models/mobilenet";
import '@tensorflow/tfjs-backend-cpu';

class Classifier {
  private readonly webcamVideo: HTMLVideoElement;
  private model: mobilenet.MobileNet;
  private readonly classifier: knnClassifier.KNNClassifier;
  constructor(webcamVideo: HTMLVideoElement) {
    this.webcamVideo = webcamVideo;
    navigator.mediaDevices
      .getUserMedia({ video: true, audio: false })
      .then((stream: MediaStream) => {
        webcamVideo.srcObject = stream;
        webcamVideo.play();
      });
    this.classifier = knnClassifier.create();
    this.loadModel();
  }

  async loadModel() {
    this.model = await mobilenet.load();
  }

  async trainClass(classId: string) {
    const img = await this.captureImage();
    const activation = this.model.infer(img, true);
    this.classifier.addExample(activation, classId);
  }

  async classify() {
    const img = await this.captureImage();
    const activation = this.model.infer(img, true);
    const result = await this.classifier.predictClass(activation);
    return result;
  }

  private async captureImage() {
    const canvas = document.createElement("canvas");
    canvas.width = this.webcamVideo.videoWidth;
    canvas.height = this.webcamVideo.videoHeight;
    const ctx = canvas.getContext("2d");
    if (ctx) {
      ctx.drawImage(this.webcamVideo, 0, 0, canvas.width, canvas.height);
    }
    return canvas;
  }
}

const cam = document.createElement("video");
const c = new Classifier(cam);

const className = document.createElement("input");
className.type = "text";
className.placeholder = "Enter A Name";
const p = document.createElement("p");
const trainButton = document.createElement("button");
trainButton.innerHTML = "Train";
trainButton.addEventListener("click", () => {
  c.trainClass(className.value);
});

const predictButton = document.createElement("button");
predictButton.innerHTML = "Predict";
async function setP() {
  p.innerHTML = (await c.classify()).label;
}
predictButton.addEventListener("click", () => {
    setP();
});
document.body.append(cam, p, trainButton, predictButton, className);
