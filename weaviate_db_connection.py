from weaviate import Client, ObjectsBatchRequest
import numpy as np

# # Define the schema
# schema = {
#     "classes": [
#         {
#             "class": "Image",
#             "properties": [
#                 {"name": "name", "dataType": ["string"]},
#                 {"name": "vector", "dataType": ["number"]}
#             ]
#         }
#     ]
# }

# # Create the schema
# client.schema.create(schema)

# # Now, let's add an image
# # First, load your image and preprocess it
# # For the sake of example, we're using a random numpy array
# img = np.random.rand(100, 100, 3)  # Replace this with your actual image loading/preprocessing
# img = img.flatten()  # Flatten the image
# img = img / np.linalg.norm(img)  # Normalize the image
# img = img.tolist()  # Convert the numpy array to a list

# # Now create a data object for the image
# data_object = {
#     "name": "my_unique_image_name2",  # Replace this with your actual unique image name
#     "vector": img
# }

# # Insert the image data object into Weaviate
# client.data_object.create(data_object, "Image2")

class_schema = client.schema.get('Image')

# result = client.query.query('''{
#     Image {
#         name
#         vector
#     }
# }''')
print(class_schema)

class WeaviateConnection:

    def __init__(self, url):
        self.client = Client(url)

        

    def create_schema(self):
        schema = {
            "classes": [
                {
                    "class": "Image",
                    "properties": [
                        {"name": "name", "dataType": ["string"]},
                        {"name": "vector", "dataType": ["number"]}
                    ]
                }
            ]
        }
        self.client.schema.create(schema)

    def add_image(self, name, vector):
        data_object = {
            "name": name,
            "vector": vector.tolist()  # flatten and normalize the numpy array to a list of floats
        }
        self.client.data_object.create(data_object, "Image")

    def get_images(self):
        result = self.client.query.graphql('''{
            Image {
                name
                vector
            }
        }''')
        return result

    def delete_image(self, image_uuid):
        self.client.data_object.delete(image_uuid)

    def add_images_batch(self, names, vectors):
        objects = ObjectsBatchRequest()
        for name, vector in zip(names, vectors):
            data_object = {
                "name": name,
                "vector": vector.tolist()  # flatten and normalize the numpy array to a list of floats
            }
            objects.add_data_object(data_object, "Image")
        self.client.batch(objects)

weaviate_connection = WeaviateConnection("http://localhost:8080")

# Create schema
weaviate_connection.create_schema()

# Add an image
weaviate_connection.add_image("image1", np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))

# Get all images
images = weaviate_connection.get_images()
print(images)

# Delete an image
weaviate_connection.delete_image("<image_uuid>")

# Add images in batch
names = ["image2", "image3"]
vectors = [np.array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1]), np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2])]
weaviate_connection.add_images_batch(names, vectors)
