from gradio_client import Client

client = Client("mrfakename/Z-Image-Turbo")

print("\n=== API DETAILS ===")
print(client.view_api())

print("\n=== END ===")