using UnityEngine;

public class GridAgentController : MonoBehaviour
{
    [SerializeField] private float moveSpeed = 2f;
    private Vector3 destination;

    private void Start() {
        destination = transform.position;
    }

    private void Update() {
        transform.position = Vector3.MoveTowards(
            transform.position,
            destination,
            moveSpeed * Time.deltaTime
        );
    }

    public void MoveToGridPosition(int x, int y) {
        destination = new Vector3(x - 2, 0.5f, 2 - y);
    }
}
