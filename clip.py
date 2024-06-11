from moviepy.editor import VideoFileClip, concatenate_videoclips

loaded_video_list = [VideoFileClip("10_часов_РАЙГОРОДСКИЙ_ТОВАРИЩ_ДРУЗЬЯ.mp4")] * 300

final_clip = concatenate_videoclips(loaded_video_list)
print(final_clip)

merged_video_name = "clip"

final_clip.write_videofile(f"{merged_video_name}.mp4")
