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

    private void Start()
    {
        agent = GetComponent<GridAgentController>();

        listener = new TcpListener(IPAddress.Loopback, 5005);
        listener.Start();

        listenerThread = new Thread(ListenForCommands)
        {
            IsBackground = true
        };
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
                    messages.Enqueue(line);
                }
            }
        }
        catch (SocketException) { }
        catch (ObjectDisposedException) { }
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

            agent.MoveToGridPosition(agentX, agentY);
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
