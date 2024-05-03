{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "67c3e663-f347-42a6-bcef-949409f3e3d9",
   "metadata": {},
   "outputs": [],
   "source": [
    "import streamlit as st\n",
    "import torch\n",
    "import torchvision.transforms as transforms\n",
    "from PIL import Image\n",
    "from forecast import forecast_image\n",
    "\n",
    "# Add other necessary imports"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "002c8e42-72b4-4673-b761-ed074312a426",
   "metadata": {},
   "outputs": [],
   "source": [
    "def main():\n",
    "        # Set the layout to three columns\n",
    "    col1, col2, col3 = st.beta_columns(3)\n",
    "    \n",
    "    # Add pictures to the first two columns\n",
    "    with col1:\n",
    "        st.image('path_to_image1.jpg', use_column_width=True)\n",
    "    \n",
    "    with col2:\n",
    "        st.image('path_to_image2.jpg', use_column_width=True)\n",
    "    \n",
    "    # Add the upload button to the middle column\n",
    "    with col3:\n",
    "        uploaded_image = st.file_uploader('Upload Image')\n",
    "        if uploaded_image is not None:\n",
    "            # Process the uploaded image\n",
    "            # Display the uploaded image\n",
    "            st.image(uploaded_image, use_column_width=True)\n",
    "    st.title(\"Predictive Ordering For Fashion Retailors Using Deep Learning\")\n",
    "\n",
    "    # File upload section\n",
    "    uploaded_file = st.file_uploader(\"Upload an image that you want to be predicted\", type=[\"jpg\", \"jpeg\", \"png\"])\n",
    "\n",
    "    if uploaded_file is not None:\n",
    "        # Read the uploaded image\n",
    "        image = Image.open(uploaded_file)\n",
    "\n",
    "        # Display the uploaded image\n",
    "        st.image(image, caption=\"Uploaded Image\", use_column_width=True)\n",
    "\n",
    "        # Perform the forecasting\n",
    "        result = forecast_image(image)\n",
    "\n",
    "        # Display the forecast result\n",
    "        st.write(\"Forecast Result:\")\n",
    "        st.write(result)\n",
    "\n",
    "if __name__ == '__main__':\n",
    "    main()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "d7728bbc-7529-484f-9fb9-ca07e86ee67e",
   "metadata": {},
   "outputs": [],
   "source": [
    "#forecast.py file for the forecasting \n",
    "def forecast_image(image):\n",
    "    # Preprocess the image\n",
    "    transform = transforms.Compose([\n",
    "        transforms.Resize((224, 224)),\n",
    "        transforms.ToTensor(),\n",
    "        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n",
    "    ])\n",
    "    input_tensor = transform(image).unsqueeze(0)\n",
    "\n",
    "    # Load the pre-trained model\n",
    "    model = load_model()  # Replace with your model loading code\n",
    "\n",
    "    # Perform the forecasting\n",
    "    result = model(input_tensor)\n",
    "\n",
    "    # Post-process the result and return the forecast\n",
    "    # Modify this part according to your specific forecasting logic\n",
    "\n",
    "    return result"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.18"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
