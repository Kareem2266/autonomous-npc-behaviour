using System.Collections;
using UnityEngine;

public class GridAgentController : MonoBehaviour
{
    [SerializeField] private float moveSpeed = 2f;
    [SerializeField] private float walkBobHeight = 0.15f;
    [SerializeField] private float walkBobFrequency = 6f;
    [SerializeField] private float bobSmoothTime = 0.12f;
    private Vector3 destination;
    private Transform[] visualParts;
    private Vector3[] visualBasePositions;
    private float walkBobPhase;
    private float currentBobOffset;
    private float bobOffsetVelocity;
    private bool wasWalking;
    private Renderer[] agentRenderers;

    private void Start() {
        destination = transform.position;
        // The root's cube mesh is only a movement placeholder. Keep the
        // child body and head as the visible character.
        MeshRenderer rootRenderer = GetComponent<MeshRenderer>();
        if (rootRenderer != null) {
            rootRenderer.enabled = false;
        }
        agentRenderers = GetComponentsInChildren<Renderer>(true);

        visualParts = new Transform[transform.childCount];
        visualBasePositions = new Vector3[transform.childCount];
        for (int i = 0; i < transform.childCount; i++) {
            visualParts[i] = transform.GetChild(i);
            visualBasePositions[i] = visualParts[i].localPosition;
        }
    }

    private void Update() {
        transform.position = Vector3.MoveTowards(
            transform.position,
            destination,
            moveSpeed * Time.deltaTime
        );

        bool isWalking = (transform.position - destination).sqrMagnitude > 0.0001f;
        if (isWalking && !wasWalking) {
            walkBobPhase = 0f;
        }
        if (isWalking) {
            walkBobPhase += Time.deltaTime * walkBobFrequency;
        }

        float targetBobOffset = isWalking
            ? (0.5f - 0.5f * Mathf.Cos(walkBobPhase)) * walkBobHeight
            : 0f;
        currentBobOffset = Mathf.SmoothDamp(
            currentBobOffset,
            targetBobOffset,
            ref bobOffsetVelocity,
            bobSmoothTime);
        wasWalking = isWalking;

        for (int i = 0; i < visualParts.Length; i++) {
            Vector3 localPosition = visualBasePositions[i];
            localPosition.y += currentBobOffset;
            visualParts[i].localPosition = localPosition;
        }
    }

    public IEnumerator PlayBathroomBreak() {
        while (!HasReachedDestination) {
            yield return null;
        }

        bool[] wasEnabled = new bool[agentRenderers.Length];
        for (int i = 0; i < agentRenderers.Length; i++) {
            wasEnabled[i] = agentRenderers[i].enabled;
            agentRenderers[i].enabled = false;
        }

        yield return new WaitForSeconds(1f);

        for (int i = 0; i < agentRenderers.Length; i++) {
            agentRenderers[i].enabled = wasEnabled[i];
        }
    }

    public bool HasReachedDestination =>
        new Vector2(transform.position.x - destination.x,
                    transform.position.z - destination.z).sqrMagnitude <= 0.0001f;

    public void MoveToGridPosition(int x, int y) {
        destination = new Vector3(x - 4.5f, 0.5f, 4.5f - y);
    }

    public void SnapToGridPosition(int x, int y) {
        destination = new Vector3(x - 4.5f, 0.5f, 4.5f - y);
        transform.position = destination;
    }
}
