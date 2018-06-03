
from PIL import Image
import os

#poke = Image.open('images/pokemon/0001.png')
#poke.show()

output_size = (28, 28)
data_dir = 'images/pokemon/'

def down_sample(data_dir):
  print('Downsampling images')

  for i, filename in enumerate(os.listdir(data_dir)):
      if i % 50 == 0:
          print(i*100/(len(os.listdir(data_dir))))
          
      img = Image.open(data_dir + filename)
      filename, f_ext = os.path.splitext(filename)

      # Downsample image size
      img.thumbnail(output_size, Image.ANTIALIAS)
      if output_size[0] == output_size[1]:
          str_size = str(output_size[0])
          
          img.save('%s/%s/~%s_%s%s' % (data_dir[:-1], str_size,
                                       filename, str_size, f_ext))
  print('Downsampling complete!')

if __name__ == "__main__":
    down_sample(data_dir)
  
