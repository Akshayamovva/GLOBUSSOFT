Task 1 : Amazon Product Scraper to CSV
Project Overview

This project allows you to scrape product information from Amazon and save it into a CSV file.
It extracts the following information for each product on a search results page:

Product Title

Product Link

Image URL

Price

Rating

Ad / Organic Status


Task 2 : Face Verification API using FastAPI + ORB
Project Overview

This project implements a face verification API using traditional computer vision techniques.
It uses OpenCV HaarCascade for face detection and ORB (Oriented FAST and Rotated BRIEF) for feature extraction.
The API accepts two face images, detects faces, extracts features, computes similarity, and returns:

Verification result: "same person" or "different person"

Similarity score

Bounding boxes of the detected faces

This method is based on classical CV and works best with clear, front-facing images. For higher accuracy, deep learning models like FaceNet, ArcFace, or DeepFace are recommended.
