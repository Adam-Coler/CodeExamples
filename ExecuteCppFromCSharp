
// Start a 45 second recording via a executable compiled in C++ that accesses the headsets internal eye tracking cameras

public static void start45SecondEyeCammeraRecording()
{
    KeyEventTimeStampDataLogger.Instance.logRow("EyeCammera", "Start45", FunctionsLibrary.time, "VarjoVideoEXEControl"); // Log that a recording is started at this time. FunctionsLibrary is a collection of scripts design to allow easy alteration of the ways that project records information. Instead of calling Time.time(format) I call the function from the library where I can set the format across the project based on what is working well when going through the data using Python.
    Process myProcess = new Process(); // Initialize a new instance of the process class
    myProcess.StartInfo.WindowStyle = ProcessWindowStyle.Normal; // set up parameters for how we want the process to behave
    myProcess.StartInfo.CreateNoWindow = false;
    myProcess.StartInfo.UseShellExecute = false;
    myProcess.StartInfo.FileName = "Path\\file name.exe"; // set up the path for the file
    myProcess.Start(); // start the process
}


// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

// This code is an edited version of the Varjo Technologies SDK
// The original code displays frames to a window
// This code instead converts them to a AVI file
// This was my first C++ code and it took some time
// Thankfully there was a solid base of code for getting the frames from the device as a png
// I had to figure out how to convert the png to a bitmap, how to record bitmaps to an avi, and how to draw on each frame

// Original: Copyright 2022 Varjo Technologies. All rights reserved.

#include "UIApplication.hpp"
#include <Globals.hpp>
#include "avi_utils.h"

#include <stdio.h>
#include <chrono>
#include <cstring>

UIApplication::UIApplication(const std::shared_ptr<Session>& session, const Options& options)
    : m_channels(options.channels)
    , m_channelCount((hasChannel(0) ? 1 : 0) + (hasChannel(1) ? 1 : 0))
    , m_stream(session, options.channels)
{
}

// add room for text to be added to each frame
int heightAddition = 60;

// set up the recording duration, tasks in the study were 30 seconds long, this covers 3 seconds prior to 2 seconds after
unsigned int videoDuration = 35;

HDC hdc;
BITMAPINFO bitmapinfo;
BITMAPINFOHEADER &bitmapinfoheader = bitmapinfo.bmiHeader;

void *bits;

HBITMAP hbitmap;
HGDIOBJ handlebitmap;
HFONT hFont;

HAVI avi;

uint8_t* temp;
uint8_t* tempBits;

time_t start, stop;

void UIApplication::setup_avi() {
    // get a device context to draw to and a generate a font
	HDC hdcscreen = GetDC(0);
	hdc = CreateCompatibleDC(hdcscreen);

	ReleaseDC(0, hdcscreen);

	hFont = CreateFont(48, 0, 0, 0, FW_DONTCARE, FALSE, TRUE, FALSE, DEFAULT_CHARSET, OUT_OUTLINE_PRECIS,
		CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, VARIABLE_PITCH, TEXT("MS Verdana"));

	SelectObject(hdc, hFont);
	SetTextColor(hdc, RGB(255, 0, 0));
}

void UIApplication::setup_bitmap() {
    // set up the header for bitmap images
    // the AVI is recorded as a series of bitmaps

	ZeroMemory(&bitmapinfo, sizeof(bitmapinfo));

    // get the size for the device stream configurations
	const auto optStreamConfig = m_stream.getConfig();
	m_streamConfig = *optStreamConfig;

	// fill out the bmiHeader for the bitmapinfo
	bitmapinfoheader.biSize = sizeof(bitmapinfoheader);
	bitmapinfoheader.biWidth = (m_streamConfig.width * m_channelCount);
	bitmapinfoheader.biHeight = (m_streamConfig.height + heightAddition);
	bitmapinfoheader.biPlanes = 1;
	bitmapinfoheader.biBitCount = 24;
	bitmapinfoheader.biCompression = BI_RGB;
	bitmapinfoheader.biSizeImage = ((bitmapinfoheader.biWidth*bitmapinfoheader.biBitCount / 8 + 3) & 0xFFFFFFFC)*bitmapinfoheader.biHeight; // 4 * bitmapinfoheader.biWidth;
	bitmapinfoheader.biXPelsPerMeter = 2835;
	bitmapinfoheader.biYPelsPerMeter = 2835;
	bitmapinfoheader.biClrUsed = 0;
	bitmapinfoheader.biClrImportant = 0;

	hbitmap = CreateDIBSection(hdc, (BITMAPINFO*)&bitmapinfoheader, DIB_RGB_COLORS, &bits, NULL, NULL);

	handlebitmap = SelectObject(hdc, hbitmap);

	tempBits = (uint8_t*)bits;
}

void UIApplication::make_avi() {

	time_t t = time(0);   // get time now
	struct tm * now = localtime(&t);

    // set up the file name for the avi
	char buffer[80];
	strftime(buffer, 80, "../35Sec_%Y_%m_%d_%H_%M_%S.avi", now);

	avi = CreateAvi(buffer, 20, NULL); // 20 at full speed

    // configure the avi options and recording settings
	AVICOMPRESSOPTIONS opts;
	ZeroMemory(&opts, sizeof(opts));
	opts.fccType = streamtypeVIDEO;
	opts.fccHandler = 1668707181; // partial

    # this will record at 200 frames a second, some compression was required
	opts.dwKeyFrameEvery = 200;
	opts.dwQuality = 9000; // 90%

	SetAviVideoCompression(avi, hbitmap, &opts, false, NULL);

	start = time(0);
}

void UIApplication::addFrame() {

    // set up a new array the size of the bitmap
    // this is set to the size of a single row in the image times the number of rows
    // but it is dealing with the sizes of the bits stored
	temp = new uint8_t[((bitmapinfoheader.biWidth*32 / 8 + 3) & 0xFFFFFFFC)*bitmapinfoheader.biHeight];

	unsigned int width_src = bitmapinfoheader.biWidth * 4;
	unsigned int width_dst = bitmapinfoheader.biWidth * 3;
	unsigned int height = bitmapinfoheader.biHeight - heightAddition;

	drawFrames(temp, m_streamConfig.width * m_channelCount * 4); // Varjo code call, this fills temp with the png frame info

    // iterate through the png frame and convert it to the bitmap format
    // This was trial and error, the first videos were upside down and blue
    // Then offset and red
    // Eventually I figured out the correct channel mappings
	for (unsigned int i = 0; i < height; ++i)
	{
		for (unsigned int j = bitmapinfoheader.biWidth; j > 0 ; --j)
		{
			tempBits[i * width_dst + j*3 + 0] = temp[(height - i) * width_src + j*4 + 0];
			tempBits[i * width_dst + j*3 + 1] = temp[(height - i) * width_src + j*4 + 1];
			tempBits[i * width_dst + j*3 + 2] = temp[(height - i) * width_src + j*4 + 2];
		}
	}

    # get time info to add to the frame
	auto timepoint = std::chrono::system_clock::now();
	auto coarse = std::chrono::system_clock::to_time_t(timepoint);
	auto fine = std::chrono::time_point_cast<std::chrono::milliseconds>(timepoint);

	CHAR buffer[sizeof "9999-12-31 23:59:59.999"];
	std::snprintf(
		buffer + strftime(buffer, sizeof buffer - 3, "%F_%T.", localtime(&coarse)),
		4,
		"%03lu",
		fine.time_since_epoch().count() % 1000);

	char buf[sizeof(uint64_t)];
	snprintf(buf, sizeof(buf), "%I64u", m_frame[0].metadata.streamFrame.frameNumber);

	char *s = new char[strlen(buffer) + strlen(buf) + 1 + 1]; //strlen(ts) + 1
	strcpy(s, buffer);
	strcat(s, "_");
	strcat(s, buf);

    // convert the text to what is required to add it to the frame
	LPCSTR out = reinterpret_cast<LPCSTR>(s);

    // add text to teh frame
	TextOutA(hdc, 0, 0, out, strlen(s));

    // add the frame to the avi
	AddAviFrame(avi, hbitmap);
}

// Additional Varjo code handled running the code and fetching the frames
// With more time I would have only recorded one channel instead of 3. The images are in greyscale so one channel could capture the same information
// while saving space
// I would have to re-work the output and write a conversion tool that took in the new outputs to convert to frames to convert to videos