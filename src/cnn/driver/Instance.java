package cnn.driver;

import java.awt.Color;
import java.awt.image.BufferedImage;

/** 
 * This is the class for each image instance.
 * @author Jonathan Groff Copyright (C) NeoCoreTechs 2022
 */
public class Instance {
	// Store the bufferedImage.
	private BufferedImage image;
	private String label;
	private String name;
	private int width, height;
	
	// Separate rgb channels.
	private int[][] red_channel, green_channel, blue_channel, gray_image;

	/** Constructs the Instance from a BufferedImage. */
	public Instance(String name, BufferedImage image, String label) {
		this.name = name;
		this.image = image;
		this.label = label;
		width = image.getWidth();
		height = image.getHeight();

		// Get separate rgb channels.
		red_channel = new int[height][width];
		green_channel = new int[height][width];
		blue_channel = new int[height][width];
		gray_image = new int[height][width];

		for (int row = 0; row < height; ++row) {
			for (int col = 0; col < width; ++col) {
				Color c = new Color(this.image.getRGB(col, row));
				red_channel[row][col] = c.getRed();
				green_channel[row][col] = c.getGreen();
				blue_channel[row][col] = c.getBlue();
			}
		}
	}
	
	/** Construct the Instance from a 3D array. */
	public Instance(int[][][] image, String label) {
		this.label = label;
		height = image[0].length;
		width = image[0][0].length;
		
		red_channel = image[0];
		green_channel = image[1];
		blue_channel = image[2];
		if (image.length == 4) {
			gray_image = image[3];
		} else {
			gray_image = new int[height][width];
			for (int i = 0; i < height; i++) {
				for (int j = 0; j < width; j++) {
					gray_image[i][j] = (image[0][i][j] + image[1][i][j] + image[2][i][j]) / 3;
				}
			}
		}
	}

	public BufferedImage getImage() {
		return image;
	}
	
	/** Gets separate red channel image. */
	public int[][] getRedChannel() {
		return red_channel;
	}

	/** Gets separate green channel image. */
	public int[][] getGreenChannel() {
		return green_channel;
	}

	/** Gets separate blue channel image. */
	public int[][] getBlueChannel() {
		return blue_channel;
	}

	/** Gets the gray scale image. */
	public int[][] getGrayImage() {
		// Gray filter
		int[] dstBuff = new int[image.getWidth()*image.getHeight()];
		readLuminance(image, dstBuff);
		for (int row = 0; row < height; ++row) {
			for (int col = 0; col < width; ++col) {
				gray_image[row][col] = dstBuff[col + row * width];
			}
		}
		return gray_image;
	}
	/**
	 * Luma represents the achromatic image while chroma represents the color component. 
	 * In video systems such as PAL, SECAM, and NTSC, a nonlinear luma component (Y') is calculated directly 
	 * from gamma-compressed primary intensities as a weighted sum, which, although not a perfect 
	 * representation of the colorimetric luminance, can be calculated more quickly without 
	 * the gamma expansion and compression used in photometric/colorimetric calculations. 
	 * In the Y'UV and Y'IQ models used by PAL and NTSC, the rec601 luma (Y') component is computed as
	 * Math.round(0.299f * r + 0.587f * g + 0.114f * b);
	 * rec601 Methods encode 525-line 60 Hz and 625-line 50 Hz signals, both with an active region covering 
	 * 720 luminance samples and 360 chrominance samples per line. The color encoding system is known as YCbCr 4:2:2.
	 * @param r
	 * @param g
	 * @param b
	 * @return Y'
	 */
	public static int luminance(float r, float g, float b) {
		return Math.round(0.299f * r + 0.587f * g + 0.114f * b);
	}
	
	/**
	 * Fill the data array with grayscale adjusted image data from sourceImage
	 */
	public static void readLuminance(BufferedImage sourceImage, int[] data) {
		int type = sourceImage.getType();
		if (type == BufferedImage.TYPE_CUSTOM || type == BufferedImage.TYPE_INT_RGB || type == BufferedImage.TYPE_INT_ARGB) {
			int[] pixels = (int[]) sourceImage.getData().getDataElements(0, 0, sourceImage.getWidth(), sourceImage.getHeight(), null);
			for (int i = 0; i < pixels.length; i++) {
				int p = pixels[i];
				int r = (p & 0xff0000) >> 16;
				int g = (p & 0xff00) >> 8;
				int b = p & 0xff;
				data[i] = luminance(r, g, b);
			}
		} else if (type == BufferedImage.TYPE_BYTE_GRAY) {
			byte[] pixels = (byte[]) sourceImage.getData().getDataElements(0, 0, sourceImage.getWidth(), sourceImage.getHeight(), null);
			for (int i = 0; i < pixels.length; i++) {
				data[i] = (pixels[i] & 0xff);
			}
		} else if (type == BufferedImage.TYPE_USHORT_GRAY) {
			short[] pixels = (short[]) sourceImage.getData().getDataElements(0, 0, sourceImage.getWidth(), sourceImage.getHeight(), null);
			for (int i = 0; i < pixels.length; i++) {
				data[i] = (pixels[i] & 0xffff) / 256;
			}
		} else if (type == BufferedImage.TYPE_3BYTE_BGR) {
            byte[] pixels = (byte[]) sourceImage.getData().getDataElements(0, 0, sourceImage.getWidth(), sourceImage.getHeight(), null);
            int offset = 0;
            int index = 0;
            for (int i = 0; i < pixels.length; i+=3) {
                int b = pixels[offset++] & 0xff;
                int g = pixels[offset++] & 0xff;
                int r = pixels[offset++] & 0xff;
                data[index++] = luminance(r, g, b);
            }
        } else {
			throw new IllegalArgumentException("Unsupported image type: " + type);
		}
	}
	
	public String getName() {
		return name;
	}
	
	/** Gets the image width. */
	public int getWidth() {
		return width;
	}

	/** Gets the image height. */
	public int getHeight() {
		return height;
	}

	/** Gets the image label. */
	public String getLabel() {
		return label;
	}
	
	public String toString() {
		return image.toString()+" "+label;
	}
}
