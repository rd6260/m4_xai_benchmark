import kagglehub

# Download latest version
path = kagglehub.competition_download('imagenet-object-localization-challenge')

print("Path to competition files:", path)
