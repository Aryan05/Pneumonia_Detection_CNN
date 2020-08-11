from flask import Flask, render_template, request
from flask_uploads import UploadSet, configure_uploads, IMAGES
import os
import glob



app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


photos = UploadSet('photos', IMAGES)
app.config['UPLOADED_PHOTOS_DEST'] = 'static/img'
configure_uploads(app, photos)

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


@app.route("/")
def index():
  #remove all old files
  files = glob.glob('static/img/*')
  visual_files = sorted(glob.glob('static/visual_img/*'))
  for f in files: 
    os.remove(f)
  for f in visual_files: 
    os.remove(f)

  return render_template('index2.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST' and 'photo' in request.files:
        filename = photos.save(request.files['photo'])
        clss, prob, pred, convert = x_ray_pred('static/img/'+filename, request.form.get('is_cnn_feat'))
        filename = 'static/img/'+filename
        if request.form.get('is_cnn_feat'):
          visual_files = sorted(glob.glob('static/visual_img/*'))
          is_cnn_feat = 'block'
        else:
          is_cnn_feat='none'
          visual_files = []
        if int(clss)==1: 
          grad_cam_disp='block'
        else: 
          grad_cam_disp='none'
          
        return render_template('report.html', pred=pred, prob=round(prob,5), filename=filename, visual_files=visual_files, convert=convert, cnn_visual=is_cnn_feat, grad_cam_disp=grad_cam_disp)

        # '<h1>Predicted Class = {}</h1><h1> Prob = {}% </h1><h1>Prediction = {}</h1>'.format(clss, prob, pred)


@app.route('/example/<example>', methods=['GET', 'POST'])
def submit_example(example):
  example_dict = {'n1':'n1.jpg', 'n2':'n2.jpeg', 'n3':'n3.jpg', 'p1':'p1.jpg', 'p2':'p2.jpg', 'p3':'p3.jpeg'}
  clss, prob, pred, convert = x_ray_pred('static/example_imgs/'+str(example_dict[example]), request.form.get('is_cnn_feat'))
  filename = '../static/example_imgs/'+str(example_dict[example])
  if int(clss)==1: 
    grad_cam_disp='block'
  else: 
    grad_cam_disp='none'

  return render_template('report.html', pred=pred, prob=round(prob,5), filename=filename, visual_files=[], convert=convert, cnn_visual='none', grad_cam_disp=grad_cam_disp)


from PIL import Image
import torch
import torchvision
from torchvision import transforms, models
import torch.nn as nn
from resnet_models import *


# model =  ResNet()
# model.load_state_dict(torch.load('model/best_model.pt', map_location='cpu'))
model = get_densenet_model()
    
model.eval()
model2 = ResNet2(model.cpu())
model2.eval()



def img_to_tensor(image_name):
  inference_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
  convert = False
  image = Image.open(image_name)
  if image.mode == 'L':
    image = image.convert(mode='RGB')
    convert=True

  image = inference_transform(image).float()
  image = image.unsqueeze(0) 
  return image, convert

def check_image(img):
  log_ps = model(img)
  ps = torch.exp(log_ps)
  top_p, top_class = ps.topk(1, dim=1)
  pred = int(top_class)
  return pred, top_p.detach().numpy().reshape(-1)[0]*100

def x_ray_pred(image, is_cnn_feat):
  
  int_to_class = ['NORMAL', 'PNEUMONIA']
  img, convert = img_to_tensor(image)
  if is_cnn_feat:
    visualize_cnn(img)
  pred, prob = check_image(img)
  if int(pred)==1:
    grad_cam(image, pred)
  
  return pred, prob, int_to_class[pred], convert

#--------------------------------------------------------
from torchvision.transforms import ToPILImage
to_img = ToPILImage()


def save_visual(output, name):
    for i in range(int(output.size(0))):
      img = to_img(output[i])
      basewidth = 150
      wpercent = (basewidth/float(img.size[0]))
      hsize = int((float(img.size[1])*float(wpercent)))
      img = img.resize((basewidth,hsize), Image.ANTIALIAS)
      img.save('static/visual_img/{}_{}.jpg'.format(name,i))

def visualize_cnn(x):
  conv1 = nn.Sequential(*list(model.features.children()))[:1](x)[0,0:10,:,:]
  layer1 = nn.Sequential(*list(model.features.children()))[:5](x)[0,:10,:,:]
  layer2 = nn.Sequential(*list(model.features.children()))[:6](x)[0,:10,:,:]

  save_visual(conv1, 'conv1')
  save_visual(layer1, 'layer1')
  save_visual(layer2, 'layer2')



def grad_cam(img_path, cls):
  files = glob.glob('static/grad_cam/*')
  for f in files: 
    os.remove(f)
  img, _ = img_to_tensor(img_path)

  pred = torch.exp(model2(img))

  pred.argmax(dim=1)
  pred[:,int(cls)].backward()
  gradients = model2.get_gradient()
  pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
  activations = model2.get_activations(img).detach()

  for i in range(512):
      activations[:, i, :, :] *= pooled_gradients[i]

  heatmap = torch.mean(activations, dim=1).squeeze()
  heatmap = np.maximum(heatmap, 0)
  heatmap /= torch.max(heatmap)
  heatmap = heatmap.numpy()

  # interpolate the heatmap
  img = cv2.imread(img_path)
  heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
  heatmap = np.uint8(255 * heatmap)
  heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
  superimposed_img = heatmap * 0.4 + img
  cv2.imwrite('static/grad_cam/map.jpg', superimposed_img)


if __name__ == "__main__":
  app.run(host= '0.0.0.0', debug=True)
