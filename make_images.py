
from PIL import Image
import os

#poke = Image.open('images/pokemon/0001.png')
#poke.show()

output_size = (28, 28)
data_dir = 'images/pokemon/sample_of_10/'
file_extension = '.bmp'

def down_sample(data_dir):
  print('Downsampling images')

  for i, filename in enumerate(os.listdir(data_dir)):
      if i % 50 == 0:
          print(i*100/(len(os.listdir(data_dir))))
      if filename.split('.')[-1] == 'png': 
        img = Image.open(data_dir + filename)
        img = alpha_to_RGB(img)

        filename, f_ext = os.path.splitext(filename)

        f_ext = file_extension

        # Downsample image size
        img.thumbnail(output_size, Image.ANTIALIAS)
        if output_size[0] == output_size[1]:
            str_size = str(output_size[0])
            #str_size in the name of the subdirectory of data_dir
            img.save('%s/%s/~%s_%s%s' % (data_dir[:-1], str_size,
                                       filename, str_size, f_ext))
  print('Downsampling complete!')

def alpha_to_RGB(png):
  # Adds background to transparent images in PNG (RGBA) format
  background = Image.new("RGB", png.size, (255, 255, 255))
  background.paste(png, mask=png.split()[3]) # 3 is the alpha channel
  return background
  
if __name__ == "__main__":
    down_sample(data_dir)
  
