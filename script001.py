# import malclient
# print(malclient.generate_token("d79a8a3b8f42750e317b0b7abc47adf2", "a9c511d6ebc3a71d16dfa5ddbdbd561b49a14fda37e0de764beeeb3e3a290ea8"))

from mal import client
from mal.enums import Field

cli = client.Client('d79a8a3b8f42750e317b0b7abc47adf2')

anime = cli.get_anime(16498, fields=Field.all_anime())
print(anime.synopsis)
print(cli.anime_fields)

