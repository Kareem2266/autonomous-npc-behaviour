using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Concurrent;
using UnityEngine;

public class PythonCommandReceiver : MonoBehaviour
{
    [SerializeField] private Transform target;
    private readonly ConcurrentQueue<string> messages = new();
    private TcpListener listener;
    private Thread listenerThread;
    private GridAgentController agent;
    private bool hasReceivedInitialState;

    private void Start()
    {
        agent = GetComponent<GridAgentController>();
        CreateStationMarker("Desk", 0, 2, new Color(0.2f, 0.65f, 0.95f));
        CreateStationMarker("Printer", 4, 2, new Color(0.95f, 0.55f, 0.15f));

        listener = new TcpListener(IPAddress.Loopback, 5005);
        listener.Start();

        listenerThread = new Thread(ListenForCommands)
        {
            IsBackground = true
        };
        listenerThread.Start();

        Debug.Log("Waiting for python on port 5005");
    }

    private void CreateStationMarker(string stationName, int x, int y, Color color)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.name = stationName;
        marker.transform.position = new Vector3(x - 2, 0.03f, 2 - y);
        marker.transform.localScale = new Vector3(0.85f, 0.06f, 0.85f);
        Destroy(marker.GetComponent<Collider>());
        marker.GetComponent<Renderer>().material.color = color;
    }

    private void ListenForCommands()
    {
        try {
            while (true)
            {
                using TcpClient client = listener.AcceptTcpClient();
                using StreamReader reader = new StreamReader(client.GetStream());

                string line;
                while ((line = reader.ReadLine()) != null)
                {
                    messages.Enqueue(line);
                }
            }
        }
        catch (SocketException) { }
        catch (ObjectDisposedException) { }
        catch (InvalidOperationException) { }
    }

    private void Update() {
         while (messages.TryDequeue(out string line)) {
            string[] values = line.Split(',');

            if (values.Length != 4 ||
                !int.TryParse(values[0], out int agentX) ||
                !int.TryParse(values[1], out int agentY) ||
                !int.TryParse(values[2], out int targetX) ||
                !int.TryParse(values[3], out int targetY)) {
                Debug.LogWarning($"Invalid position message: {line}");
                continue;
            }

            if (!hasReceivedInitialState) {
                agent.SnapToGridPosition(agentX, agentY);
                hasReceivedInitialState = true;
            } else {
                agent.MoveToGridPosition(agentX, agentY);
            }
            target.position = new Vector3(
            targetX - 2,
            target.position.y,
            2 - targetY);
        }
    }

    private void OnDestroy()
    {
        listener?.Stop();
    }
}
