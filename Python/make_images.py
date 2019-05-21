
from PIL import Image
import os

#poke = Image.open('images/pokemon/0001.png')
#poke.show()

use_color = False
output_size = (28, 28)
file_extension = '.png'
display_before_save = True

input_dir = 'images/pokemon/sample_of_10/'

str_size = str(output_size[0])
output_subdir = input_dir[:-1]+'/'
if use_color is False:
  output_subdir += 'monochrome/'
output_subdir += str_size

def down_sample(data_dir):
  print('Downsampling images')

  for i, filename in enumerate(os.listdir(input_dir)):
      if i % 50 == 0:
          print(i*100/(len(os.listdir(input_dir))))
      if filename.split('.')[-1] == 'png': 
        img = Image.open(data_dir + filename)
        img = alpha_to_RGB(img)

        if use_color is False:
          img = to_black_and_white(img)

        filename, f_ext = os.path.splitext(filename)

        f_ext = file_extension

        # Downsample image size
        img.thumbnail(output_size, Image.ANTIALIAS)
        if output_size[0] == output_size[1]:
            str_size = str(output_size[0])
            #str_size in the name of the subdirectory of data_dir
            if display_before_save and i < 3: img.show()
            img.save('%s/%s_%s%s' % (output_subdir, filename, str_size, f_ext))
  print('Downsampling complete!')

def alpha_to_RGB(png):
  # Adds background to transparent images in PNG (RGBA) format
  background = Image.new("RGB", png.size, (255, 255, 255))
  background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
  return background

def to_black_and_white(img):
  return img.convert('L')
  
  
if __name__ == "__main__":
    down_sample(input_dir)
  
