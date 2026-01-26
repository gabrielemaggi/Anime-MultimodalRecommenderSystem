import pandas as pd
import requests

path = "dataset/AnimeList.csv"

# load the dataset and print the head
dataset = pd.read_csv(path)
print(dataset.head())

# taking the ids and the cdn link for
# the thumbnail
ids = dataset["anime_id"]
links = dataset["image_url"]

# search for null data
print(f"There are null ids? {dataset["anime_id"].isnull().any()}")
print(f"There are null links? {dataset['image_url'].isnull().any()}")
print("-"*50)

# highlighting null data
image_urls_index = []
null_links = dataset["image_url"].isnull()

for index in range(0, len(ids)):
    if null_links[index]:
        image_urls_index.append(index)

print(f"Number of null links: {len(image_urls_index)}")
print("-"*50)
print(f"index | anime_id | image_url")
for index in image_urls_index:
    print(f"{index} | {dataset.loc[index]["anime_id"]} | {dataset.loc[index]["image_url"]}")
print("-"*50)

# drop null data
dataset.drop(image_urls_index, inplace=True)
print(f"There are null links? {dataset['image_url'].isnull().any()}")
print("-"*50)

# link correction and image download
link = "cdn.myanimelist.net"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

for index in dataset.index:

    # get the url and fix with the right link
    url = dataset.loc[index, "image_url"]
    dataset.loc[index, "image_url"] = url.replace("myanimelist.cdn-dena.com", link)

    # download the image
    try:
        new_url = dataset.loc[index, "image_url"]
        response = requests.get(new_url, headers=headers, stream=True)

        # image name
        file_name = "dataset/images/" + str(dataset.loc[index, "anime_id"]) + ".jpg"

        # saving the image in the directory
        if response.status_code == 200:
            with open(file_name, 'wb') as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
        else:
            print(f"Error while downloading thumbnail at index {index}. Status code: {response.status_code}")

    except Exception as e:
        print(f"Error: {e}")




