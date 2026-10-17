using System;
using System.Collections;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Threading;
using System.Collections.Concurrent;
using UnityEngine;

public class PythonCommandReceiver : MonoBehaviour
{
    [SerializeField] private Transform target;
    [SerializeField] private GameObject deskPrefab;
    [SerializeField] private GameObject printerStationPrefab;
    [SerializeField] private GameObject snackStationPrefab;
    [SerializeField] private GameObject coffeeStationPrefab;
    [SerializeField] private GameObject bathroomStationPrefab;
    private bool isSecondaryWorker;
    private readonly ConcurrentQueue<string> messages = new();
    private TcpListener listener;
    private Thread listenerThread;
    private GridAgentController agent;
    private GridAgentController[] agents = new GridAgentController[2];
    private Transform[] obstacleMarkers = new Transform[2];
    private Transform snackItem;
    private Transform[] coffeeCupParts = Array.Empty<Transform>();
    private Transform printedPaper;
    private Transform bathroomStation;
    private Transform deskStation;
    private Transform printerStation;
    private Transform snackStationTransform;
    private Transform coffeeStationTransform;
    private Vector3 printerPaperStart;
    private Vector3 printerPaperEnd;
    private bool hasReceivedInitialState;

    private void Start()
    {
        if (isSecondaryWorker)
        {
            return;
        }

        agent = GetComponent<GridAgentController>();
        agents[0] = agent;
        agent.SnapToGridPosition(0, 0);
        CreateOfficeDecor();
        if (snackStationPrefab == null)
        {
            snackStationPrefab = Resources.Load<GameObject>("SnackStation");
        }
        if (coffeeStationPrefab == null)
        {
            coffeeStationPrefab = Resources.Load<GameObject>("CoffeeStation");
        }
        if (bathroomStationPrefab == null)
        {
            bathroomStationPrefab = Resources.Load<GameObject>("PortaPotty");
        }
        if (printerStationPrefab == null)
        {
            printerStationPrefab = Resources.Load<GameObject>("PrinterStation");
        }
        GameObject spawnedBathroom = SpawnStationPrefab(
            bathroomStationPrefab, "Bathroom", new Vector3(3.5f, 0f, -3.5f));
        if (spawnedBathroom != null)
        {
            bathroomStation = spawnedBathroom.transform;
        }
        GameObject snackStation = SpawnStationPrefab(
            snackStationPrefab, "Snack Area", new Vector3(-1.5f, 0f, 2.5f));
        if (snackStation != null)
        {
            snackStationTransform = snackStation.transform;
            DecorateSnackStation(snackStation.transform);
            snackItem = snackStation.transform.Find("Snack Bag");
        }

        GameObject coffeeStation = SpawnStationPrefab(
            coffeeStationPrefab, "Coffee Area", new Vector3(-1.5f, 0f, -3.5f));
        if (coffeeStation != null)
        {
            coffeeStationTransform = coffeeStation.transform;
            DecorateCoffeeStation(coffeeStation.transform);
            coffeeCupParts = new[]
            {
                coffeeStation.transform.Find("Coffee Cup Lower"),
                coffeeStation.transform.Find("Coffee Cup Rim")
            };
        }
        GameObject printerStation = SpawnStationPrefab(
            printerStationPrefab, "Printer", new Vector3(3.5f, 0f, 2.5f));
        if (printerStation != null)
        {
            this.printerStation = printerStation.transform;
            DecoratePrinterStation(printerStation.transform);
            printedPaper = printerStation.transform.Find("Printed Paper");
            if (printedPaper != null)
            {
                printerPaperStart = printedPaper.localPosition;
                printerPaperEnd = printerPaperStart + Vector3.back * 0.32f;
            }
        }
        if (deskPrefab != null)
        {
            GameObject spawnedDesk = Instantiate(
                deskPrefab,
                new Vector3(-3.5f, 0f, 0.5f),
                Quaternion.identity);
            spawnedDesk.name = "Desk at Worker 1 Station";
            deskStation = spawnedDesk.transform;
            DecorateDesk(spawnedDesk.transform);
            Debug.Log($"Desk spawned at {spawnedDesk.transform.position}");

            GameObject secondDesk = Instantiate(
                deskPrefab,
                new Vector3(-3.5f, 0f, -1.5f),
                Quaternion.identity);
            secondDesk.name = "Desk at Worker 2 Station";
            DecorateDesk(secondDesk.transform);
        }
        else
        {
            Debug.LogWarning("Desk prefab is not assigned on PythonCommandReceiver.");
        }
        obstacleMarkers[0] = CreateObstacleMarker("Obstacle 1");
        obstacleMarkers[1] = CreateObstacleMarker("Obstacle 2");

        // The trained policy controls two workers. Reuse this scene worker's
        // visual/controller setup for the second worker, without a second listener.
        GameObject secondWorker = Instantiate(agent.gameObject);
        secondWorker.name = "Agent 2";
        PythonCommandReceiver duplicateReceiver =
            secondWorker.GetComponent<PythonCommandReceiver>();
        if (duplicateReceiver != null)
        {
            duplicateReceiver.isSecondaryWorker = true;
        }
        agents[1] = secondWorker.GetComponent<GridAgentController>();
        agents[1].SetCharacterPrefab(Resources.Load<GameObject>("Secretary"));
        agents[1].SnapToGridPosition(0, 1);

        listener = new TcpListener(IPAddress.Loopback, 5005);
        listener.Start();

        listenerThread = new Thread(ListenForCommands)
        {
            IsBackground = true
        };
        listenerThread.Start();

        Debug.Log("Waiting for python on port 5005");
    }

    private GameObject SpawnStationPrefab(GameObject prefab, string stationName, Vector3 position)
    {
        if (prefab == null)
        {
            Debug.LogWarning($"{stationName} prefab is not assigned on PythonCommandReceiver.");
            return null;
        }

        GameObject station = Instantiate(prefab, position, Quaternion.identity);
        station.name = stationName;
        Debug.Log($"{stationName} spawned at {station.transform.position}");
        return station;
    }

    private void CreateOfficeDecor()
    {
        GameObject decorRoot = new GameObject("Office Finish Details");
        Transform parent = decorRoot.transform;

        Material teal = CreateOfficeMaterial(new Color(0.20f, 0.38f, 0.39f), 0.18f);
        Material trim = CreateOfficeMaterial(new Color(0.87f, 0.82f, 0.70f), 0.15f);
        Material frame = CreateOfficeMaterial(new Color(0.12f, 0.22f, 0.27f), 0.3f);
        Material glass = CreateOfficeMaterial(new Color(0.42f, 0.67f, 0.72f), 0.72f);
        Material signFace = CreateOfficeMaterial(new Color(0.16f, 0.28f, 0.31f), 0.22f);

        Renderer floor = GameObject.Find("OfficeFloor")?.GetComponent<Renderer>();
        if (floor != null)
        {
            floor.material.color = new Color(0.68f, 0.70f, 0.67f);
        }

        // Low wall panels and trim make the three-sided room feel finished
        // while keeping the open front clear for the camera.
        CreateOfficeCube("Back Wainscot", new Vector3(0f, 0.29f, 4.84f),
            new Vector3(9.72f, 0.58f, 0.12f), teal, parent);
        CreateOfficeCube("Left Wainscot", new Vector3(-4.84f, 0.29f, 0f),
            new Vector3(0.12f, 0.58f, 9.72f), teal, parent);
        CreateOfficeCube("Right Wainscot", new Vector3(4.84f, 0.29f, 0f),
            new Vector3(0.12f, 0.58f, 9.72f), teal, parent);

        CreateOfficeCube("Back Chair Rail", new Vector3(0f, 0.62f, 4.76f),
            new Vector3(9.8f, 0.07f, 0.16f), trim, parent);
        CreateOfficeCube("Left Chair Rail", new Vector3(-4.76f, 0.62f, 0f),
            new Vector3(0.16f, 0.07f, 9.8f), trim, parent);
        CreateOfficeCube("Right Chair Rail", new Vector3(4.76f, 0.62f, 0f),
            new Vector3(0.16f, 0.07f, 9.8f), trim, parent);

        // Two framed windows add depth and a daylight cue to the back wall.
        CreateOfficeWindow(-2.65f, frame, glass, trim, parent);
        CreateOfficeWindow(2.65f, frame, glass, trim, parent);

        CreateOfficeCube("Office Sign Backing", new Vector3(0f, 1.76f, 4.78f),
            new Vector3(2.35f, 0.40f, 0.10f), signFace, parent);

        // Recolor the existing key light to a softer warm-neutral tone.
        Light keyLight = FindAnyObjectByType<Light>();
        if (keyLight != null && keyLight.type == LightType.Directional)
        {
            keyLight.color = new Color(1f, 0.94f, 0.84f);
            keyLight.intensity = 1.35f;
            keyLight.shadowStrength = 0.55f;
        }
    }

    private void CreateOfficeWindow(float centerX, Material frame, Material glass,
        Material trim, Transform parent)
    {
        const float centerY = 1.30f;
        const float centerZ = 4.78f;
        const float width = 2.05f;
        const float height = 0.92f;
        const float border = 0.09f;

        CreateOfficeCube("Window Frame", new Vector3(centerX, centerY, centerZ),
            new Vector3(width + border * 2f, height + border * 2f, 0.10f), frame, parent);
        CreateOfficeCube("Window Glass", new Vector3(centerX, centerY, centerZ - 0.07f),
            new Vector3(width, height, 0.025f), glass, parent);
        CreateOfficeCube("Window Sill", new Vector3(centerX, centerY - height / 2f - 0.07f,
            centerZ - 0.11f), new Vector3(width + border * 2.6f, 0.10f, 0.22f), trim, parent);
        CreateOfficeCube("Window Center Bar", new Vector3(centerX, centerY, centerZ - 0.10f),
            new Vector3(0.045f, height, 0.035f), trim, parent);
        CreateOfficeCube("Window Crossbar", new Vector3(centerX, centerY, centerZ - 0.10f),
            new Vector3(width, 0.045f, 0.035f), trim, parent);
    }

    private Material CreateOfficeMaterial(Color color, float smoothness)
    {
        Shader shader = Shader.Find("Universal Render Pipeline/Lit");
        if (shader == null)
        {
            shader = Shader.Find("Standard");
        }

        Material material = new Material(shader);
        material.color = color;
        if (material.HasProperty("_Smoothness"))
        {
            material.SetFloat("_Smoothness", smoothness);
        }
        return material;
    }

    private void CreateOfficeCube(string objectName, Vector3 position, Vector3 scale,
        Material material, Transform parent)
    {
        GameObject piece = GameObject.CreatePrimitive(PrimitiveType.Cube);
        piece.name = objectName;
        piece.transform.SetParent(parent, false);
        piece.transform.position = position;
        piece.transform.localScale = scale;
        piece.GetComponent<Renderer>().sharedMaterial = material;
        Destroy(piece.GetComponent<Collider>());
    }

    private void DecorateDesk(Transform desk)
    {
        Material charcoal = CreateOfficeMaterial(new Color(0.12f, 0.15f, 0.17f), 0.28f);
        Material screen = CreateOfficeMaterial(new Color(0.18f, 0.43f, 0.50f), 0.48f);
        Material chairFabric = CreateOfficeMaterial(new Color(0.16f, 0.29f, 0.32f), 0.18f);
        Material paper = CreateOfficeMaterial(new Color(0.94f, 0.91f, 0.82f), 0.05f);
        Material keycaps = CreateOfficeMaterial(new Color(0.55f, 0.58f, 0.55f), 0.12f);

        // Monitor, stand, and keyboard sit on the existing desktop.
        CreateDeskPiece("Monitor Base", desk, new Vector3(-0.10f, 0.80f, 0.15f),
            new Vector3(0.22f, 0.025f, 0.12f), charcoal);
        CreateDeskPiece("Monitor Stand", desk, new Vector3(-0.10f, 0.86f, 0.15f),
            new Vector3(0.045f, 0.12f, 0.045f), charcoal);
        CreateDeskPiece("Monitor", desk, new Vector3(-0.10f, 1.00f, 0.15f),
            new Vector3(0.40f, 0.29f, 0.055f), charcoal);
        CreateDeskPiece("Monitor Screen", desk, new Vector3(-0.10f, 1.00f, 0.118f),
            new Vector3(0.35f, 0.235f, 0.012f), screen);
        CreateDeskPiece("Keyboard", desk, new Vector3(0.02f, 0.81f, -0.12f),
            new Vector3(0.34f, 0.035f, 0.13f), charcoal);

        for (int row = 0; row < 3; row++)
        {
            for (int column = 0; column < 8; column++)
            {
                float keyX = -0.115f + column * 0.031f;
                float keyZ = -0.163f + row * 0.034f;
                CreateDeskPiece("Keyboard Key", desk,
                    new Vector3(keyX, 0.833f, keyZ),
                    new Vector3(0.023f, 0.008f, 0.020f), keycaps);
            }
        }

        CreateDeskPiece("Paper Stack", desk, new Vector3(0.26f, 0.795f, 0.13f),
            new Vector3(0.16f, 0.018f, 0.20f), paper);
        CreateDeskPiece("Top Sheet", desk, new Vector3(0.245f, 0.808f, 0.12f),
            new Vector3(0.15f, 0.008f, 0.18f), paper);

        // Tuck a small chair under the camera-facing edge of the desk.
        CreateDeskPiece("Chair Seat", desk, new Vector3(0f, 0.43f, -0.43f),
            new Vector3(0.38f, 0.09f, 0.34f), chairFabric);
        CreateDeskPiece("Chair Back", desk, new Vector3(0f, 0.72f, -0.62f),
            new Vector3(0.38f, 0.48f, 0.08f), chairFabric);
        CreateDeskPiece("Chair Stem", desk, new Vector3(0f, 0.23f, -0.43f),
            new Vector3(0.07f, 0.33f, 0.07f), charcoal);
        CreateDeskPiece("Chair Base Front Spoke", desk, new Vector3(0f, 0.06f, -0.59f),
            new Vector3(0.05f, 0.035f, 0.27f), charcoal);
        CreateDeskPiece("Chair Base Rear Spoke", desk, new Vector3(0f, 0.06f, -0.27f),
            new Vector3(0.05f, 0.035f, 0.27f), charcoal);
        CreateDeskPiece("Chair Base Left Spoke", desk, new Vector3(-0.16f, 0.06f, -0.43f),
            new Vector3(0.27f, 0.035f, 0.05f), charcoal);
        CreateDeskPiece("Chair Base Right Spoke", desk, new Vector3(0.16f, 0.06f, -0.43f),
            new Vector3(0.27f, 0.035f, 0.05f), charcoal);
        CreateDeskPiece("Chair Caster", desk, new Vector3(0f, 0.035f, -0.72f),
            new Vector3(0.07f, 0.07f, 0.07f), charcoal);
        CreateDeskPiece("Chair Caster", desk, new Vector3(0f, 0.035f, -0.14f),
            new Vector3(0.07f, 0.07f, 0.07f), charcoal);
        CreateDeskPiece("Chair Caster", desk, new Vector3(-0.29f, 0.035f, -0.43f),
            new Vector3(0.07f, 0.07f, 0.07f), charcoal);
        CreateDeskPiece("Chair Caster", desk, new Vector3(0.29f, 0.035f, -0.43f),
            new Vector3(0.07f, 0.07f, 0.07f), charcoal);
    }

    private void DecoratePrinterStation(Transform printer)
    {
        Material bezel = CreateOfficeMaterial(new Color(0.11f, 0.15f, 0.17f), 0.3f);
        Material display = CreateOfficeMaterial(new Color(0.23f, 0.62f, 0.66f), 0.5f);
        Material buttonGreen = CreateOfficeMaterial(new Color(0.31f, 0.63f, 0.44f), 0.18f);
        Material buttonAmber = CreateOfficeMaterial(new Color(0.88f, 0.58f, 0.22f), 0.18f);

        CreateDeskPiece("Printer Display Bezel", printer, new Vector3(0.12f, 0.94f, -0.211f),
            new Vector3(0.12f, 0.08f, 0.018f), bezel);
        CreateDeskPiece("Printer Display Glass", printer, new Vector3(0.12f, 0.94f, -0.222f),
            new Vector3(0.09f, 0.052f, 0.008f), display);
        CreateDeskPiece("Printer Green Button", printer, new Vector3(-0.12f, 0.86f, -0.211f),
            new Vector3(0.045f, 0.035f, 0.012f), buttonGreen);
        CreateDeskPiece("Printer Amber Button", printer, new Vector3(-0.05f, 0.86f, -0.211f),
            new Vector3(0.045f, 0.035f, 0.012f), buttonAmber);
    }

    private void DecorateCoffeeStation(Transform coffee)
    {
        Material bezel = CreateOfficeMaterial(new Color(0.12f, 0.16f, 0.17f), 0.3f);
        Material display = CreateOfficeMaterial(new Color(0.32f, 0.66f, 0.65f), 0.42f);
        Material button = CreateOfficeMaterial(new Color(0.84f, 0.57f, 0.26f), 0.18f);
        Material saucer = CreateOfficeMaterial(new Color(0.88f, 0.84f, 0.73f), 0.22f);

        CreateDeskPiece("Coffee Machine Control Bezel", coffee, new Vector3(0f, 0.99f, -0.197f),
            new Vector3(0.13f, 0.09f, 0.018f), bezel);
        CreateDeskPiece("Coffee Machine Display", coffee, new Vector3(0f, 0.99f, -0.209f),
            new Vector3(0.095f, 0.055f, 0.008f), display);
        CreateDeskPiece("Coffee Machine Button", coffee, new Vector3(0.105f, 0.91f, -0.197f),
            new Vector3(0.045f, 0.045f, 0.012f), button);
        CreateDeskPiece("Coffee Cup Saucer", coffee, new Vector3(-0.23f, 0.77f, 0.17f),
            new Vector3(0.20f, 0.02f, 0.20f), saucer);
        CreateDeskPiece("Coffee Cup Handle", coffee, new Vector3(-0.32f, 0.88f, 0.17f),
            new Vector3(0.045f, 0.075f, 0.035f), saucer);
    }

    private void DecorateSnackStation(Transform snacks)
    {
        Material label = CreateOfficeMaterial(new Color(0.96f, 0.87f, 0.64f), 0.12f);

        CreateDeskPiece("Snack Bag Front Label", snacks, new Vector3(-0.23f, 0.86f, -0.164f),
            new Vector3(0.105f, 0.045f, 0.012f), label);
        CreateDeskPiece("Snack Box Front Label", snacks, new Vector3(0.23f, 0.86f, -0.164f),
            new Vector3(0.105f, 0.045f, 0.012f), label);
    }

    private void CreateDeskPiece(string objectName, Transform desk, Vector3 localPosition,
        Vector3 localScale, Material material)
    {
        GameObject piece = GameObject.CreatePrimitive(PrimitiveType.Cube);
        piece.name = objectName;
        piece.transform.SetParent(desk, false);
        piece.transform.localPosition = localPosition;
        piece.transform.localScale = localScale;
        piece.GetComponent<Renderer>().sharedMaterial = material;
        Destroy(piece.GetComponent<Collider>());
    }

    private IEnumerator AnimatePickup(Transform[] items)
    {
        Vector3[] startingPositions = new Vector3[items.Length];
        for (int i = 0; i < items.Length; i++)
        {
            if (items[i] == null)
            {
                continue;
            }
            startingPositions[i] = items[i].localPosition;
        }

        const float duration = 0.65f;
        const float liftHeight = 0.22f;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float lift = liftHeight * Mathf.Sin(Mathf.PI * Mathf.Clamp01(elapsed / duration));
            for (int i = 0; i < items.Length; i++)
            {
                if (items[i] != null)
                {
                    items[i].localPosition = startingPositions[i] + Vector3.up * lift;
                }
            }
            yield return null;
        }

        for (int i = 0; i < items.Length; i++)
        {
            if (items[i] != null)
            {
                items[i].localPosition = startingPositions[i];
            }
        }
    }

    private IEnumerator AnimatePrintedPaper()
    {
        if (printedPaper == null)
        {
            yield break;
        }

        printedPaper.gameObject.SetActive(true);
        printedPaper.localPosition = printerPaperStart;
        const float duration = 0.7f;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.SmoothStep(0f, 1f, Mathf.Clamp01(elapsed / duration));
            printedPaper.localPosition = Vector3.Lerp(printerPaperStart, printerPaperEnd, t);
            yield return null;
        }
        printedPaper.localPosition = printerPaperEnd;
        yield return new WaitForSeconds(0.5f);
        printedPaper.gameObject.SetActive(false);
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
            if (line.StartsWith("MULTI,")) {
                ApplyMultiAgentState(line);
                continue;
            }

            if (line.StartsWith("ROUTE:")) {
                string route = line.Substring("ROUTE:".Length).Replace(",", " -> ");
                Debug.Log($"Planned task route: {route}");
                continue;
            }

            if (line.StartsWith("INTERACTION:")) {
                string interaction = line.Substring("INTERACTION:".Length);
                HandleInteraction(0, interaction);
                continue;
            }

            if (line.StartsWith("EVENT,"))
            {
                string[] eventParts = line.Split(',');
                if (eventParts.Length == 3 &&
                    int.TryParse(eventParts[1], out int workerIndex) &&
                    workerIndex >= 0 && workerIndex < agents.Length)
                {
                    HandleInteraction(workerIndex, eventParts[2]);
                }
                else
                {
                    Debug.LogWarning($"Invalid policy event: {line}");
                }
                continue;
            }

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

    private void ApplyMultiAgentState(string line)
    {
        string[] values = line.Split(',');
        // MULTI, x1,y1,gx1,gy1, x2,y2,gx2,gy2, ox1,oy1,ox2,oy2
        if (values.Length != 13)
        {
            Debug.LogWarning($"Invalid multi-agent state: {line}");
            return;
        }

        int[] state = new int[12];
        for (int i = 0; i < state.Length; i++)
        {
            if (!int.TryParse(values[i + 1], out state[i]))
            {
                Debug.LogWarning($"Invalid multi-agent state: {line}");
                return;
            }
        }

        for (int i = 0; i < agents.Length; i++)
        {
            int offset = i * 4;
            if (!hasReceivedInitialState)
            {
                agents[i].SnapToGridPosition(state[offset], state[offset + 1]);
            }
            else
            {
                agents[i].MoveToGridPosition(state[offset], state[offset + 1]);
            }
            if (i == 0)
            {
                target.position = new Vector3(
                    state[offset + 2] - 4.5f,
                    target.position.y,
                    4.5f - state[offset + 3]);
            }
        }

        hasReceivedInitialState = true;
        for (int i = 0; i < obstacleMarkers.Length; i++)
        {
            int offset = 8 + i * 2;
            obstacleMarkers[i].position = new Vector3(
                state[offset] - 4.5f,
                0.4f,
                4.5f - state[offset + 1]);
        }
    }

    private void HandleInteraction(int workerIndex, string interaction)
    {
        Debug.Log($"Worker {workerIndex + 1} station interaction: {interaction}");
        GridAgentController worker = agents[workerIndex];
        Transform interactingStation = GetStationForInteraction(interaction);
        if (interactingStation != null)
        {
            worker.FaceTargetDuringInteraction(interactingStation);
        }

        if (interaction.Equals("SNACK", StringComparison.OrdinalIgnoreCase) ||
            interaction.IndexOf("snack", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            StartCoroutine(AnimatePickup(new[] { snackItem }));
        }
        else if (interaction.Equals("COFFEE", StringComparison.OrdinalIgnoreCase) ||
                 interaction.IndexOf("coffee", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            StartCoroutine(AnimatePickup(coffeeCupParts));
        }
        else if (interaction.Equals("PRINTER", StringComparison.OrdinalIgnoreCase) ||
                 interaction.IndexOf("printer", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            StartCoroutine(AnimatePrintedPaper());
        }
        else if (interaction.Equals("BATHROOM", StringComparison.OrdinalIgnoreCase) ||
                 interaction.IndexOf("bathroom", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            StartCoroutine(worker.PlayBathroomBreak());
            StartCoroutine(AnimateBathroomJump(worker));
        }
    }

    private Transform GetStationForInteraction(string interaction)
    {
        if (interaction.IndexOf("printer", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return printerStation;
        }
        if (interaction.IndexOf("bathroom", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return bathroomStation;
        }
        if (interaction.IndexOf("snack", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return snackStationTransform;
        }
        if (interaction.IndexOf("coffee", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return coffeeStationTransform;
        }
        if (interaction.IndexOf("desk", StringComparison.OrdinalIgnoreCase) >= 0)
        {
            return deskStation;
        }
        return null;
    }

    private IEnumerator AnimateBathroomJump(GridAgentController worker)
    {
        if (bathroomStation == null)
        {
            yield break;
        }

        while (!worker.HasReachedDestination)
        {
            yield return null;
        }

        Vector3 startPosition = bathroomStation.localPosition;
        const float duration = 1f;
        const float jumpHeight = 0.45f;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.deltaTime;
            float lift = jumpHeight * Mathf.Sin(
                Mathf.PI * Mathf.Clamp01(elapsed / duration));
            bathroomStation.localPosition = startPosition + Vector3.up * lift;
            yield return null;
        }

        bathroomStation.localPosition = startPosition;
    }

    private void OnDestroy()
    {
        listener?.Stop();
    }
}
