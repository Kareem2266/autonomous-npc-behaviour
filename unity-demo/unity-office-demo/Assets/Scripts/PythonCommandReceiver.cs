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
    private Transform[] obstacleMarkers = new Transform[2];
    private bool hasReceivedInitialState;

    private void Start()
    {
        agent = GetComponent<GridAgentController>();
        CreateStationMarker("Desk", 1, 5, new Color(0.2f, 0.65f, 0.95f));
        CreateStationMarker("Printer", 8, 2, new Color(0.95f, 0.55f, 0.15f));
        CreateStationMarker("Bathroom", 8, 8, new Color(0.65f, 0.4f, 0.8f));
        CreateStationMarker("Snack Area", 3, 2, new Color(0.3f, 0.75f, 0.35f));
        CreateStationMarker("Coffee Area", 3, 8, new Color(0.55f, 0.32f, 0.18f));
        obstacleMarkers[0] = CreateObstacleMarker("Obstacle 1");
        obstacleMarkers[1] = CreateObstacleMarker("Obstacle 2");

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
        marker.transform.position = new Vector3(x - 4.5f, 0.03f, 4.5f - y);
        marker.transform.localScale = new Vector3(0.85f, 0.06f, 0.85f);
        Destroy(marker.GetComponent<Collider>());
        marker.GetComponent<Renderer>().material.color = color;

        GameObject labelObject = new GameObject($"{stationName} Label");
        labelObject.transform.position = new Vector3(
            x - 4.5f,
            0.067f,
            4.5f - y);
        labelObject.transform.rotation = Quaternion.Euler(90f, 0f, 0f);

        TextMesh label = labelObject.AddComponent<TextMesh>();
        label.text = stationName;
        label.anchor = TextAnchor.MiddleCenter;
        label.alignment = TextAlignment.Center;
        label.fontSize = 64;
        label.characterSize = 0.2f / stationName.Length;
        label.fontStyle = FontStyle.Bold;
        label.color = Color.black;

        Shader depthTestedTextShader = Shader.Find("Sprites/Default");
        if (depthTestedTextShader != null)
        {
            Material textMaterial = new Material(depthTestedTextShader);
            textMaterial.mainTexture = label.font.material.mainTexture;
            textMaterial.color = Color.black;
            label.GetComponent<MeshRenderer>().material = textMaterial;
        }
    }

    private Transform CreateObstacleMarker(string obstacleName)
    {
        GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cube);
        marker.name = obstacleName;
        marker.transform.localScale = new Vector3(0.85f, 0.8f, 0.85f);
        Destroy(marker.GetComponent<Collider>());
        marker.GetComponent<Renderer>().material.color = new Color(0.25f, 0.25f, 0.25f);
        return marker.transform;
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

            if (values.Length != 8 ||
                !int.TryParse(values[0], out int agentX) ||
                !int.TryParse(values[1], out int agentY) ||
                !int.TryParse(values[2], out int targetX) ||
                !int.TryParse(values[3], out int targetY) ||
                !int.TryParse(values[4], out int obstacleOneX) ||
                !int.TryParse(values[5], out int obstacleOneY) ||
                !int.TryParse(values[6], out int obstacleTwoX) ||
                !int.TryParse(values[7], out int obstacleTwoY)) {
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
            targetX - 4.5f,
            target.position.y,
            4.5f - targetY);
            obstacleMarkers[0].position = new Vector3(
                obstacleOneX - 4.5f,
                0.4f,
                4.5f - obstacleOneY);
            obstacleMarkers[1].position = new Vector3(
                obstacleTwoX - 4.5f,
                0.4f,
                4.5f - obstacleTwoY);
        }
    }

    private void OnDestroy()
    {
        listener?.Stop();
    }
}
