import Algorithmia

input = "https://pbs.twimg.com/profile_images/714630467409018884/2ywNrMx2.jpg"
client = Algorithmia.client('simmp0NmxBIAkbVwazmgI8QQvMg1')
algo = client.algo('sfw/NudityDetection/1.1.4')
print algo.pipe(input)
