import os
import PIL.Image



def convertjpg(pngfile,outdir,width=32,height=32):
    img = PIL.Image.open('C:/Users/cjl/Desktop/1500/'+pngfile)
    try:
        new_img = img.resize((width, height), PIL.Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(pngfile)))
    except Exception as e:
        print(e)
for pngfile in os.listdir('C:/Users/cjl/Desktop/1500'):
    print(pngfile)
    convertjpg(pngfile, "C:/Users/cjl/Desktop/test32")