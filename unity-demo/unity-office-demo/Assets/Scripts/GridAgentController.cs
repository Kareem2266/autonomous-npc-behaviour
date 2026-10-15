using System.Collections;
using UnityEngine;

public class GridAgentController : MonoBehaviour
{
    // Temporary comment: remove after Unity refreshes this script.
    [SerializeField] private float moveSpeed = 2f;
    [SerializeField] private float walkCycleLength = 1.1f;
    [SerializeField] private float legSwingAngle = 28f;
    [SerializeField] private float armSwingAngle = 22f;

    private Vector3 destination;
    private float gaitPhase;
    private Renderer[] agentRenderers;
    private Transform characterVisualRoot;
    private Transform leftLegPivot;
    private Transform rightLegPivot;
    private Transform leftArmPivot;
    private Transform rightArmPivot;
    private Transform interactionTarget;
    private bool isInteracting;

    private void Start()
    {
        destination = transform.position;

        // Keep the grid-controlled Agent as an invisible movement root.
        Renderer[] existingRenderers = GetComponentsInChildren<Renderer>(true);
        foreach (Renderer existingRenderer in existingRenderers)
        {
            existingRenderer.enabled = false;
        }

        Transform oldHead = transform.Find("AgentHead");
        if (oldHead != null)
        {
            oldHead.gameObject.SetActive(false);
        }

        Transform oldBody = transform.Find("RobotBody");
        if (oldBody != null)
        {
            oldBody.gameObject.SetActive(false);
        }

        GameObject characterAsset = Resources.Load<GameObject>("MiiCharacter");
        if (characterAsset == null)
        {
            Debug.LogError("MiiCharacter.obj was not imported from Assets/Resources.", this);
            agentRenderers = new Renderer[0];
            return;
        }

        characterVisualRoot = new GameObject("Mii Character").transform;
        characterVisualRoot.SetParent(transform, false);
        characterVisualRoot.localPosition = new Vector3(0f, -0.45f, 0f);
        characterVisualRoot.localRotation = Quaternion.identity;

        GameObject character = Instantiate(characterAsset, characterVisualRoot, false);
        character.name = "Mii Character Mesh";

        GameObject legAsset = Resources.Load<GameObject>("MiiLeg");
        if (legAsset == null)
        {
            Debug.LogError("MiiLeg.obj was not imported from Assets/Resources.", this);
        }
        else
        {
            leftLegPivot = CreateLegPivot("Left Leg Pivot", -0.12f, legAsset);
            rightLegPivot = CreateLegPivot("Right Leg Pivot", 0.12f, legAsset);
        }

        GameObject armAsset = Resources.Load<GameObject>("MiiArm");
        if (armAsset == null)
        {
            Debug.LogError("MiiArm.obj was not imported from Assets/Resources.", this);
        }
        else
        {
            leftArmPivot = CreateArmPivot("Left Arm Pivot", -0.28f, armAsset);
            rightArmPivot = CreateArmPivot("Right Arm Pivot", 0.28f, armAsset);
        }

        agentRenderers = GetComponentsInChildren<Renderer>(true);
    }

    private void Update()
    {
        Vector3 previousPosition = transform.position;
        transform.position = Vector3.MoveTowards(
            transform.position,
            destination,
            moveSpeed * Time.deltaTime
        );

        float movedDistance = Vector3.Distance(previousPosition, transform.position);
        if (movedDistance > 0.0001f)
        {
            // Advance the gait by distance, so each stride cycles forward and back
            // consistently even when the NPC pauses or changes direction.
            float safeCycleLength = Mathf.Max(0.1f, walkCycleLength);
            gaitPhase += movedDistance * (Mathf.PI * 2f / safeCycleLength);
        }

        bool isWalking = (transform.position - destination).sqrMagnitude > 0.0001f;
        Vector3 facingDirection = isWalking
            ? destination - transform.position
            : isInteracting && interactionTarget != null
                ? interactionTarget.position - transform.position
                : Vector3.zero;
        facingDirection.y = 0f;
        if (characterVisualRoot != null && facingDirection.sqrMagnitude > 0.0001f)
        {
            // The imported model faces local -Z, so negate the look direction
            // for Unity's +Z-facing LookRotation convention.
            Quaternion facing = Quaternion.LookRotation(-facingDirection.normalized);
            characterVisualRoot.localRotation = Quaternion.Slerp(
                characterVisualRoot.localRotation,
                facing,
                8f * Time.deltaTime
            );
        }

        float swing = isWalking ? Mathf.Sin(gaitPhase) * legSwingAngle : 0f;
        AnimateLimb(leftLegPivot, swing);
        AnimateLimb(rightLegPivot, -swing);
        float armSwing = isWalking ? Mathf.Sin(gaitPhase) * armSwingAngle : 0f;
        AnimateLimb(leftArmPivot, -armSwing);
        AnimateLimb(rightArmPivot, armSwing);
    }

    private Transform CreateLegPivot(string pivotName, float xPosition, GameObject legAsset)
    {
        Transform pivot = new GameObject(pivotName).transform;
        pivot.SetParent(characterVisualRoot, false);
        pivot.localPosition = new Vector3(xPosition, 0.52f, 0f);
        GameObject leg = Instantiate(legAsset, pivot, false);
        leg.name = pivotName.Replace(" Pivot", " Mesh");
        return pivot;
    }

    private Transform CreateArmPivot(string pivotName, float xPosition, GameObject armAsset)
    {
        Transform pivot = new GameObject(pivotName).transform;
        pivot.SetParent(characterVisualRoot, false);
        pivot.localPosition = new Vector3(xPosition, 0.99f, 0f);
        GameObject arm = Instantiate(armAsset, pivot, false);
        arm.name = pivotName.Replace(" Pivot", " Mesh");
        return pivot;
    }

    private void AnimateLimb(Transform legPivot, float angle)
    {
        if (legPivot == null)
        {
            return;
        }

        Quaternion targetRotation = Quaternion.Euler(angle, 0f, 0f);
        legPivot.localRotation = Quaternion.Slerp(
            legPivot.localRotation,
            targetRotation,
            12f * Time.deltaTime
        );
    }

    public IEnumerator PlayBathroomBreak()
    {
        while (!HasReachedDestination)
        {
            yield return null;
        }

        bool[] wasEnabled = new bool[agentRenderers.Length];
        for (int i = 0; i < agentRenderers.Length; i++)
        {
            wasEnabled[i] = agentRenderers[i].enabled;
            agentRenderers[i].enabled = false;
        }

        yield return new WaitForSeconds(1f);

        for (int i = 0; i < agentRenderers.Length; i++)
        {
            agentRenderers[i].enabled = wasEnabled[i];
        }
    }

    public bool HasReachedDestination =>
        new Vector2(transform.position.x - destination.x,
                    transform.position.z - destination.z).sqrMagnitude <= 0.0001f;

    public void FaceTargetDuringInteraction(Transform targetTransform)
    {
        interactionTarget = targetTransform;
        isInteracting = true;
    }

    public void MoveToGridPosition(int x, int y)
    {
        isInteracting = false;
        destination = new Vector3(x - 4.5f, 0.5f, 4.5f - y);
    }

    public void SnapToGridPosition(int x, int y)
    {
        isInteracting = false;
        destination = new Vector3(x - 4.5f, 0.5f, 4.5f - y);
        transform.position = destination;
    }
}
