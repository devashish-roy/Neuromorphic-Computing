import dv_processing as dv
import logging
import sys
import cv2 as cv
import matplotlib.pyplot as plt

logger = logging.getLogger()
date_time_string_format = '%Y-%m-%y %H:%M:%S'
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt=date_time_string_format
)

def load_data_from(aedat4_file):
    """
    Load data from an Aedat4 file.
    :param aedat4_file: path to the Aedat4 file as a string.
    :return: A reader containing event, frame, imu and trigger streams within the supplied aedat4 file.
    """
    reader = dv.io.MonoCameraRecording(aedat4_file)
    if not reader.isEventStreamAvailable():
        logger.critical("Something went wrong. The camera data does not have an event stream available.\n")

    return reader

def event_stream_resolution(camera_data):
    """
    Reads the resolution of the event stream within the supplied camera data reader.
    :param camera_data: An aedat4 file reader.
    :return: width and height resolution tuple.
    """
    resolution = camera_data.getEventResolution()
    return resolution

def get_events_from(data):
    """
    Extract the event stream from the supplied data reader.
    :param data: An aedat4 file reader.
    :return: Events contained in the supplied data reader.
    """
    events = dv.EventStore()

    while data.isRunning():
        temp_event_store = data.getNextEventBatch()
        if temp_event_store is not None:
            events.add(temp_event_store)

    return events

def _calculate_crop_origin(center=(173, 130), width=100, height=100):
    origin_x = center[0] - width / 2
    origin_y = center[1] - height / 2
    return int(origin_x), int(origin_y)

def crop_area(
        aedat4_file,
        center,
        crop_width = 100,
        crop_height = 100
):
    """
    Crop the area of the event stream within the supplied aedat4 file to a rectangular central area.
    :param aedat4_file: path to the aedat4 file as a string.
    :param center: a tuple with the x and y coordinates of the center of the cropped area.
    :param crop_width: horizontal length of the cropped area.
    :param crop_height: vertical length of the cropped area.
    :return: An event store containing the events within the central area.
    """
    crop_origin = _calculate_crop_origin(center, crop_width, crop_height)
    region_filter = dv.EventRegionFilter((*crop_origin, crop_width, crop_height))
    region_filter.accept(get_events_from(load_data_from(aedat4_file)))
    filtered_events = region_filter.generateEvents()
    return filtered_events

def crop_area_all_event_streams(
        aedat4_file,
        center,
        crop_width = 100,
        crop_height = 100
):
    """
    Crop the area of the event stream within the supplied aedat4 file to a rectangular central area.
    :param aedat4_file: path to the aedat4 file as a string.
    :param center: a tuple with the x and y coordinates of the center of the cropped area.
    :param crop_width: horizontal length of the cropped area.
    :param crop_height: vertical length of the cropped area.
    :return: An event store containing all the event streams within the central area.
    """
    crop_origin = _calculate_crop_origin(center, crop_width, crop_height)
    region_filter = dv.EventRegionFilter((*crop_origin, crop_width, crop_height))
    cropped_event_streams = dv.EventStore()
    
    data = load_data_from(aedat4_file) # Loading data      
    
    while data.isRunning():        
        temp_event_store = data.getNextEventBatch()
        if temp_event_store is not None:
            region_filter.accept(temp_event_store)
            filtered_events = region_filter.generateEvents()
            if filtered_events.size() > 0:
                cropped_event_streams.add(filtered_events)    
    
    return cropped_event_streams

def events_info(events):
    """
    Relevant information about the given events.
    :param events: An event store
    :return: A dictionary with keys: duration, first timestamp, last time stamp and events count of the given events.
    """
    return {
        'duration': events.duration(),
        'first timestamp': events.timestamps()[0],
        'last timestamp': events.timestamps()[-1],
        'events count': events.size()
    }

def crop_preview_area(
        aedat4_file,
        center,
        crop_width = 100,
        crop_height = 100
):
    """
    Presents a screenshot of the original event stream and the cropped area side by side.
    :param aedat4_file: path to the Aedat4 file as a string.
    :param center: a tuple with the x and y coordinates of the center of the cropped area.
    :param crop_width: horizontal length of the cropped area.
    :param crop_height: vertical length of the cropped area.
    :return: None
    """
    data = load_data_from(aedat4_file)
    source_resolution = event_stream_resolution(data)
    source_events = get_events_from(data)
    filtered_events = crop_area(aedat4_file, center, crop_width, crop_height)

    visualizer = dv.visualization.EventVisualizer(source_resolution)
    visualizer_input = visualizer.generateImage(source_events)
    visualizer_output = visualizer.generateImage(filtered_events)

    preview = cv.hconcat([visualizer_input, visualizer_output])
    """
    cv.namedWindow("preview", cv.WINDOW_NORMAL)
    cv.imshow("preview", preview)
    cv.waitKey(0)
    cv.destroyAllWindows()
    """
    # Display in Colab using Matplotlib
    plt.figure(figsize=(10, 5))
    plt.title("Original (Left) vs Cropped (Right)")
    plt.imshow(cv.cvtColor(preview, cv.COLOR_BGR2RGB))  # Convert to RGB for Matplotlib
    plt.axis('off')
    plt.show()

def events_to_aedat4_file(
        events,
        resolution = (346, 260),
        file_name = 'cropped.aedat4'
) -> None:
    """
    Saves the given events to an aedat4 file.
    :param events: An event store
    :param resolution: A tuple specifying the resolution (width and height in pixels) of the given events.
    :param file_name: The file name of the generated aedat4 file.
    :return: None
    """
    config = dv.io.MonoCameraWriter.EventOnlyConfig(cameraName="DAVIS346_00000305", resolution=resolution)
    writer = dv.io.MonoCameraWriter(file_name, config)
    writer.writeEvents(events)