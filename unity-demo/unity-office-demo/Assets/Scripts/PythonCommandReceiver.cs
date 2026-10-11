using System;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Concurrent;
using UnityEngine;

public class PythonCommandReceiver : MonoBehaviour
{
    private readonly ConcurrentQueue<Vector2Int> positions = new();
    private TcpListener listener;
    private Thread listenerThread;
    private GridAgentController agent;

    private void Start()
    {
        agent = GetComponent<GridAgentController>();

        listener = new TcpListener(IPAddress.Loopback, 5005);
        listener.Start();

        listenerThread = new Thread(ListenForCommands);
        listenerThread.IsBackground = true;
        listenerThread.Start();

        Debug.Log("Waiting for python on port 5005");
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
                    string[] coordinates = line.Split(',');

                    if (coordinates.Length == 2 &&
                        int.TryParse(coordinates[0], out int x) &&
                        int.TryParse(coordinates[1], out int y))
                    {
                        positions.Enqueue(new Vector2Int(x, y));
                    }
                }
            }
        }
        catch (SocketException) { }
        catch (ObjectDisposedException) { }
    }

    private void Update()
    {
        while (positions.TryDequeue(out Vector2Int position))
        {
            agent.MoveToGridPosition(position.x, position.y);
        }
    }

    private void OnDestroy()
    {
        listener?.Stop();
    }
}
