
# this code is part of a class called "KeyEventTimeStampDataLogger"

public static KeyEventTimeStampDataLogger Instance { get; private set; } // setting up a singleton design

private void OnEnable() // called when this object becomes active
{
    if (Instance != null) { Instance = this; } // if there is no instance of this class, this becomes the static instance
    else { Destroy(Instance); Instance = this; } // if there is an instance, destroy it, then this becomes the static instance

    AffectElicitationStudyManager.startLoggingData += setUpDataLogging; // subscribe to an event that triggers when the environment is ready to start data logging
}

private void OnDisable() // called when this object is disabled
{
    AffectElicitationStudyManager.startLoggingData -= setUpDataLogging; // unsubscribe from the event that triggers logging
}

private async void setUpDataLogging() // when the setUpDataLogging event is triggered we initialize a new CSV writer object, this initialization is threaded so the method is async
{
    if (fileID == "") { fileID = this.name; } // confirm that there is a file name for the CSV, if there is not, name it after the object this script is attached to

    // Participant identifiers are held in a separate singleton class that should be one of the first active objects
    // This code is checking that there is an instance of that class, if there is use the ID that it has, otherwise the ID is null
    PID = AffectElicitationStudyManager.Instance ? AffectElicitationStudyManager.Instance.participantID : "Null";

    fileID = PID + "_" + fileID; // set up a file name following the naming convention used in this project

    m_CSVWritter = gameObject.AddComponent<CSVWritter>(); // instantiate a new CSV writer object and attach it to the same object as this
    m_CSVWritter.Initalize(fileID, header, "PID_" + PID); // Initialize the CSV writer with the file name, column names for what will be recorded (header), and the ID
    await m_CSVWritter.MakeNewSession(); // start a thread that makes a new file and wait for thread execution to finish.
    m_CSVWritter.StartNewCSV(); // start the new CSV logging by recording the header information.

    isLogging = true; // Set a flag that this script is ready to log
}

# this function is called from other classes to log data to the csv recorded by this class
public void logRow(string eventType, string eventName, string eventTime, string senderName)
{
    if (isLogging)
    {
        string[] rowContent = {
            PID,
            gameObject.name,
            FunctionsLibrary.time,
            eventType,
            eventName,
            eventTime,
            senderName
        };
        m_CSVWritter.AddRow(rowContent);
    }
}