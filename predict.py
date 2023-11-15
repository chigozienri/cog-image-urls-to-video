# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md

import asyncio
import json
import os
import subprocess
import tempfile
from typing import Optional
from zipfile import ZipFile

import aiohttp
from cog import BaseModel, BasePredictor, Input, Path


class Output(BaseModel):
    video: Path
    zip: Optional[Path]


async def is_valid_url(session, url):
    try:
        async with session.head(url) as response:
            response.raise_for_status()
            return True
    except aiohttp.ClientError:
        return False


async def download_and_save_image(session, url, temp_dir, frame_number):
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            image_data = await response.read()

            # Use tempfile to create a temporary file in the specified directory
            _, file_extension = os.path.splitext(url)
            temp_file_name = f"frame_{frame_number}{file_extension}"
            temp_file_path = os.path.join(temp_dir, temp_file_name)

            # Save the image to the temporary file
            with open(temp_file_path, "wb") as f:
                f.write(image_data)

            return temp_file_path
    except aiohttp.ClientError as e:
        print(f"Error downloading image from {url}: {e}")
        return None


async def parse_json_string_and_save_images(json_string):
    try:
        parsed_data = json.loads(json_string)

        # Check if the parsed data is a list
        if not isinstance(parsed_data, list):
            raise ValueError("The parsed data is not a list.")

        async with aiohttp.ClientSession() as session:
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()

            # Check if all elements in the list are valid URLs and download/save images
            tasks = [
                download_and_save_image(session, url, temp_dir, frame_number)
                for frame_number, url in enumerate(parsed_data)
                if await is_valid_url(session, url)
            ]

            saved_images = await asyncio.gather(*tasks)

            return saved_images, temp_dir

    except (ValueError, json.JSONDecodeError) as e:
        print(f"Error parsing JSON string: {e}")
        return None, None


def create_animated_media(images, output_filename, fps, mp4=False):
    try:
        # Use ffmpeg to create an animated gif or video from the images
        command = [
            "ffmpeg",
            "-r",
            str(fps),
            "-pattern_type",
            "glob",
            "-i",
            os.path.join(os.path.dirname(images[0]), "frame_*.png"),
        ]

        if mp4:
            command += [
                "-pix_fmt",
                "yuv420p",
                "-c:v",
                "libx264",
                "-movflags",
                "faststart",
                "-qp",
                "17",
            ]
        else:
            command += ["-pix_fmt", "rgb8"]

        command += [output_filename]
        subprocess.check_output(command)
        return output_filename
    except subprocess.CalledProcessError as e:
        print(f"Error creating animated media: {e}")
        return None


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # Nothing to set up!

    def predict(
        self,
        image_urls: str = Input(description="A list of input urls"),
        mp4: bool = Input(
            description="Returns .mp4 if true or .gif if false", default=False
        ),
        fps: float = Input(description="Frames per second of output video", default=4),
        output_zip: bool = Input(
            description="Also returns a zip of the input images if true", default=False
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        saved_images, temp_dir = asyncio.run(
            parse_json_string_and_save_images(image_urls)
        )
        if saved_images and temp_dir:
            output_filename = "animated.mp4" if mp4 else "animated.gif"
            if os.path.exists(output_filename):
                os.remove(output_filename)
            create_animated_media(saved_images, output_filename, fps, mp4=mp4)
            output = Output(video=Path(output_filename))
            if output_zip:
                zip_filename = "inputs.zip"
                if os.path.exists(zip_filename):
                    os.remove(zip_filename)
                with ZipFile(zip_filename, "w") as zip:
                    for file_path in Path(temp_dir).rglob("*"):
                        zip.write(file_path, arcname=file_path.relative_to(temp_dir))
                output.zip = Path(zip_filename)
        return output
