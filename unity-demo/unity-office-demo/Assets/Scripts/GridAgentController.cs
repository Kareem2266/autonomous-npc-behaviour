using System.Collections;
using UnityEngine;

public class GridAgentController : MonoBehaviour
{
    [SerializeField] private float moveSpeed = 2f;
    [SerializeField] private float walkCycleLength = 1.25f;
    [SerializeField] private float legSwingDegrees = 12f;
    [SerializeField] private float armSwingDegrees = 8f;

    private Vector3 destination;
    private float gaitPhase;
    private bool isWalking;
    private Renderer[] agentRenderers;
    private Transform characterVisualRoot;
    private Transform interactionTarget;
    private bool isInteracting;

    private Transform leftUpperLeg;
    private Transform rightUpperLeg;
    private Transform leftUpperArm;
    private Transform rightUpperArm;
    private Transform leftLowerArm;
    private Transform rightLowerArm;
    private Quaternion leftUpperLegRestRotation;
    private Quaternion rightUpperLegRestRotation;
    private Quaternion leftUpperArmRestRotation;
    private Quaternion rightUpperArmRestRotation;

    private void Start()
    {
        destination = transform.position;

        // Keep the grid-controlled Agent as an invisible movement root.
        foreach (Renderer existingRenderer in GetComponentsInChildren<Renderer>(true))
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

        GameObject officeWorkerPrefab = Resources.Load<GameObject>("OfficeWorker");
        if (officeWorkerPrefab == null)
        {
            Debug.LogError("OfficeWorker.prefab was not found in Assets/Resources.", this);
            agentRenderers = new Renderer[0];
            return;
        }

        characterVisualRoot = new GameObject("Office Worker Visual").transform;
        characterVisualRoot.SetParent(transform, false);
        // The grid root is centered half a unit above the floor; the imported
        // character's pivot is at its feet, so offset it back down to floor level.
        characterVisualRoot.localPosition = new Vector3(0f, -0.5f, 0f);
        characterVisualRoot.localRotation = Quaternion.identity;

        GameObject character = Instantiate(officeWorkerPrefab, characterVisualRoot, false);
        character.name = "Office Worker";

        Animator animator = character.GetComponentInChildren<Animator>();
        if (animator != null && animator.avatar != null && animator.avatar.isHuman)
        {
            animator.applyRootMotion = false;
            // Use the imported bind pose as the idle pose. The pack has no
            // animation clips, so animate the humanoid bones directly below.
            animator.enabled = false;
            leftUpperLeg = animator.GetBoneTransform(HumanBodyBones.LeftUpperLeg);
            rightUpperLeg = animator.GetBoneTransform(HumanBodyBones.RightUpperLeg);
            leftUpperArm = animator.GetBoneTransform(HumanBodyBones.LeftUpperArm);
            rightUpperArm = animator.GetBoneTransform(HumanBodyBones.RightUpperArm);
            leftLowerArm = animator.GetBoneTransform(HumanBodyBones.LeftLowerArm);
            rightLowerArm = animator.GetBoneTransform(HumanBodyBones.RightLowerArm);

            if (leftUpperLeg != null) leftUpperLegRestRotation = leftUpperLeg.localRotation;
            if (rightUpperLeg != null) rightUpperLegRestRotation = rightUpperLeg.localRotation;
            if (leftUpperArm != null) leftUpperArmRestRotation = leftUpperArm.localRotation;
            if (rightUpperArm != null) rightUpperArmRestRotation = rightUpperArm.localRotation;
        }
        else
        {
            Debug.LogError("OfficeWorker needs a valid Humanoid Avatar to animate.", character);
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
            gaitPhase += movedDistance * (Mathf.PI * 2f / Mathf.Max(0.1f, walkCycleLength));
        }

        isWalking = (transform.position - destination).sqrMagnitude > 0.0001f;
        Vector3 facingDirection = isWalking
            ? destination - transform.position
            : isInteracting && interactionTarget != null
                ? interactionTarget.position - transform.position
                : Vector3.zero;
        facingDirection.y = 0f;

        if (characterVisualRoot != null && facingDirection.sqrMagnitude > 0.0001f)
        {
            Quaternion facing = Quaternion.LookRotation(facingDirection.normalized);
            characterVisualRoot.localRotation = Quaternion.Slerp(
                characterVisualRoot.localRotation,
                facing,
                8f * Time.deltaTime
            );
        }
    }

    private void LateUpdate()
    {
        if (characterVisualRoot == null)
        {
            return;
        }

        float stride = isWalking ? Mathf.Sin(gaitPhase) : 0f;
        Vector3 characterRight = characterVisualRoot.right;

        // Reset to the imported rest pose each frame, then add only a small
        // forward/back swing while walking. At rest, the legs stay straight.
        if (leftUpperLeg != null)
        {
            leftUpperLeg.localRotation = leftUpperLegRestRotation;
            leftUpperLeg.rotation = Quaternion.AngleAxis(
                stride * legSwingDegrees,
                characterRight
            ) * leftUpperLeg.rotation;
        }
        if (rightUpperLeg != null)
        {
            rightUpperLeg.localRotation = rightUpperLegRestRotation;
            rightUpperLeg.rotation = Quaternion.AngleAxis(
                -stride * legSwingDegrees,
                characterRight
            ) * rightUpperLeg.rotation;
        }

        PoseArm(leftUpperArm, leftLowerArm, leftUpperArmRestRotation, -1f, stride);
        PoseArm(rightUpperArm, rightLowerArm, rightUpperArmRestRotation, 1f, -stride);
    }

    private void PoseArm(
        Transform upperArm,
        Transform lowerArm,
        Quaternion restRotation,
        float side,
        float stride
    )
    {
        if (upperArm == null || lowerArm == null)
        {
            return;
        }

        upperArm.localRotation = restRotation;
        Vector3 currentArmDirection = (lowerArm.position - upperArm.position).normalized;
        Vector3 outward = characterVisualRoot.right * side;
        Vector3 relaxedDirection = (Vector3.down * 0.995f + outward * 0.05f).normalized;
        upperArm.rotation = Quaternion.FromToRotation(currentArmDirection, relaxedDirection)
            * upperArm.rotation;
        upperArm.rotation = Quaternion.AngleAxis(
            stride * armSwingDegrees,
            characterVisualRoot.right
        ) * upperArm.rotation;
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
