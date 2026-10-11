using UnityEngine;

public class GridAgentController : MonoBehaviour
{
    public void MoveToGridPosition(int x, int y) {
        transform.position = new Vector3(x - 2, 0.5f, 2 - y);
    }
}
