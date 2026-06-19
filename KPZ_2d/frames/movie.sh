ffmpeg -framerate 20 \
  -pattern_type glob \
  -i "frame_*.png" \
  -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" \
  -c:v libx264 \
  -crf 18 \
  -preset slow \
  -pix_fmt yuv420p \
  KPZ_Sphere.mp4
